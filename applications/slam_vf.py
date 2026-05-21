"""
SLAM-VF: Visual SLAM-style workload over ordered video frames.

The orchestrator sends frames one-by-one via gRPC (base64-encoded JPEG bytes),
in dataset order, waiting for each response before sending the next frame
(same request/response pattern as image_processing and object_detection,
but with cross-frame state).

Per-session state kept on the edge:
  * previous frame's grayscale image, ORB keypoints and descriptors
  * accumulated 4x4 camera pose (world <- current camera)
  * total keyframes inserted
  * total map points accumulated (a coarse, per-keyframe count of inliers)
  * monotonic frame counter (state_version)
  * compact ring buffer of keyframe records used in checkpoints

Each frame:
  1. decode JPEG -> grayscale
  2. ORB feature detection + descriptor extraction
  3. brute-force Hamming match against previous frame's descriptors
  4. essential matrix + recoverPose -> incremental (R, t)
  5. compose into the running 4x4 pose
  6. promote to keyframe when translation magnitude exceeds threshold

Checkpointing: metadata only. Raw frames, descriptors, and gray images are
NEVER serialized. The snapshot carries video_id, state_version, last frame id,
the running pose (decomposed to x/y/theta), keyframe + map-point counters, and
a compact ring buffer of recent keyframe records.
"""

import logging
import math
import os
import threading
import time
import base64
from collections import deque

import numpy as np
import cv2
import grpc

from utils import current_milli_time
from protos import benchmark_pb2 as pb2
from protos import benchmark_pb2_grpc as pb2_grpc
import configs


# --------------------------------------------------------------------------- #
# Per-session state
# --------------------------------------------------------------------------- #
class _SessionState:
    __slots__ = (
        "prev_gray", "prev_kps", "prev_desc",
        "pose", "num_keyframes", "num_map_points",
        "version", "last_frame_id", "keyframes",
        "log_tail", "req_history", "last_checkpoint_version",
    )

    def __init__(self):
        self.prev_gray = None
        self.prev_kps = None
        self.prev_desc = None
        self.pose = np.eye(4, dtype=np.float32)
        self.num_keyframes = 0
        self.num_map_points = 0
        # ``version`` is the per-session frame counter (current_state_version).
        # ``last_checkpoint_version`` is the version captured by the most recent
        # checkpoint exposed via to_snapshot(); together they let the
        # orchestrator decide between FULL and INCREMENTAL recovery and trim
        # the log tail at the right watermark.
        self.version = 0
        self.last_checkpoint_version = 0
        self.last_frame_id = -1
        history = getattr(configs, "SLAM_VF_CHECKPOINT_KEYFRAME_HISTORY", 50)
        # Each entry: (frame_id, tx, ty, tz, theta, num_inliers).
        self.keyframes = deque(maxlen=max(1, int(history)))
        # Per-frame log tail used for incremental recovery (metadata only).
        # Entry: (version, frame_id, tracked, num_features, num_matches,
        #         num_inliers, was_keyframe, pose_x, pose_y, pose_theta).
        log_max = getattr(configs, "SLAM_VF_LOG_TAIL_MAX", 2000)
        self.log_tail = deque(maxlen=max(1, int(log_max)))
        # Bounded recent-req-id cache for ACTIVE-mode idempotency. Membership
        # check is O(N) on the deque; N is small (~1000) so this is fine.
        rid_max = getattr(configs, "SLAM_VF_REQ_ID_CACHE_MAX", 1000)
        self.req_history = deque(maxlen=max(1, int(rid_max)))


_lock = threading.RLock()
_sessions = {}  # session_id -> _SessionState

_orb = cv2.ORB_create(nfeatures=getattr(configs, "SLAM_VF_NUM_FEATURES", 1000))
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# --------------------------------------------------------------------------- #
# Replication / role state (mirrors stateful_sentiment_aggregator)
# --------------------------------------------------------------------------- #
ROLE = pb2.ROLE_BACKUP
PEER_HOSTS = []
EDGE_GRPC_PORT = int(os.getenv("EDGE_GRPC_PORT", "50051"))

_checkpoint_period = 30
_checkpoint_thread = None
_checkpoint_thread_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_session(session_id, reset):
    with _lock:
        if reset or session_id not in _sessions:
            _sessions[session_id] = _SessionState()
        return _sessions[session_id]


def _decode_frame_to_gray(b64_frame):
    raw = base64.b64decode(bytearray(b64_frame, encoding="utf-8"))
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("SLAM-VF: failed to decode frame")
    return img


def _estimate_pose(prev_kps, prev_desc, kps, desc):
    """Match descriptors and recover incremental pose."""
    if prev_desc is None or desc is None or len(prev_kps) < 8 or len(kps) < 8:
        return None, None, 0, 0

    matches = _matcher.match(prev_desc, desc)
    if len(matches) < 8:
        return None, None, len(matches), 0

    pts_prev = np.float32([prev_kps[m.queryIdx].pt for m in matches])
    pts_curr = np.float32([kps[m.trainIdx].pt for m in matches])

    K = np.array([[700.0, 0.0, 320.0],
                  [0.0, 700.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    E, mask = cv2.findEssentialMat(pts_prev, pts_curr, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None, None, len(matches), 0

    inliers, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, K, mask=mask)
    return R, t, len(matches), int(inliers)


def _compose(pose_4x4, R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.reshape(3).astype(np.float32)
    return (pose_4x4 @ T).astype(np.float32)


def _decompose_pose_xytheta(pose):
    """Top-down monocular trajectory projection: X-Z plane + yaw about Y."""
    P = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    tx, _ty, tz = float(P[0, 3]), float(P[1, 3]), float(P[2, 3])
    theta = math.atan2(float(P[0, 2]), float(P[0, 0]))
    return tx, tz, theta


# --------------------------------------------------------------------------- #
# Main frame processing
# --------------------------------------------------------------------------- #
def _build_duplicate_response(session, request, request_received_time_ms):
    """Build a no-state-change response for a request whose req_id was already
    applied. Returns the session's current cumulative state so ACTIVE-mode
    callers (FIRST / MAJORITY) see consistent values regardless of which
    replica actually applied the original request."""
    resp = pb2.SLAMVFResponse()
    resp.session_id = request.session_id
    resp.frame_id = request.frame_id
    resp.tracked = False
    resp.num_features = 0
    resp.num_matches = 0
    resp.num_inliers = 0
    resp.num_map_points = session.num_map_points
    resp.num_keyframes = session.num_keyframes
    resp.pose.extend([float(v) for v in session.pose.reshape(-1).tolist()])
    resp.state_version = session.version
    resp.frame_bytes_size = len(request.frame) if request.frame is not None else 0
    resp.applied = False
    resp.request_time_ms = request.request_time_ms
    resp.request_received_time_ms = request_received_time_ms
    resp.response_time_ms = current_milli_time()
    return resp


def process_frame(request, request_received_time_ms):
    session = _get_session(request.session_id, request.reset)

    # Idempotency: under ACTIVE replication the same (session_id, frame_id,
    # req_id) tuple may be retried. Short-circuit duplicates BEFORE decoding
    # the frame so we don't pay ORB cost for them.
    req_id = getattr(request, "req_id", "") or ""
    if req_id and not request.reset:
        with _lock:
            if req_id in session.req_history:
                return _build_duplicate_response(session, request,
                                                 request_received_time_ms)

    gray = _decode_frame_to_gray(request.frame)
    kps, desc = _orb.detectAndCompute(gray, None)

    tracked = False
    num_matches = 0
    num_inliers = 0

    with _lock:
        # Second dedup check inside the lock: another thread may have applied
        # the same req_id concurrently while we were decoding / running ORB.
        if req_id and req_id in session.req_history:
            return _build_duplicate_response(session, request,
                                             request_received_time_ms)
        was_keyframe = False
        if session.prev_desc is not None:
            R, t, num_matches, num_inliers = _estimate_pose(
                session.prev_kps, session.prev_desc, kps, desc
            )
            if R is not None and t is not None and num_inliers >= 8:
                session.pose = _compose(session.pose, R, t)
                tracked = True
                if float(np.linalg.norm(t)) > getattr(
                        configs, "SLAM_VF_KEYFRAME_TRANS_THRESH", 0.05):
                    session.num_keyframes += 1
                    session.num_map_points += num_inliers
                    was_keyframe = True
                    _tx, _ty, theta = _decompose_pose_xytheta(session.pose)
                    session.keyframes.append((
                        int(request.frame_id),
                        float(session.pose[0, 3]),
                        float(session.pose[1, 3]),
                        float(session.pose[2, 3]),
                        float(theta),
                        int(num_inliers),
                    ))

        session.prev_gray = gray
        session.prev_kps = kps
        session.prev_desc = desc
        session.version += 1
        session.last_frame_id = int(request.frame_id)

        pose_flat = session.pose.reshape(-1).tolist()
        num_features = 0 if kps is None else len(kps)
        num_keyframes = session.num_keyframes
        num_map_points = session.num_map_points
        state_version = session.version

        # Append one log-tail entry per processed frame for incremental recovery.
        tx_now, ty_now, theta_now = _decompose_pose_xytheta(session.pose)
        session.log_tail.append((
            int(state_version),
            int(request.frame_id),
            bool(tracked),
            int(num_features),
            int(num_matches),
            int(num_inliers),
            bool(was_keyframe),
            float(tx_now),
            float(ty_now),
            float(theta_now),
        ))

        # Record applied req_id last so dedup is observable from the next call.
        if req_id:
            session.req_history.append(req_id)

    resp = pb2.SLAMVFResponse()
    resp.session_id = request.session_id
    resp.frame_id = request.frame_id
    resp.tracked = tracked
    resp.num_features = num_features
    resp.num_matches = num_matches
    resp.num_inliers = int(num_inliers)
    resp.num_map_points = num_map_points
    resp.num_keyframes = num_keyframes
    resp.pose.extend([float(v) for v in pose_flat])
    resp.state_version = state_version
    resp.frame_bytes_size = len(request.frame) if request.frame is not None else 0
    resp.applied = True
    resp.request_time_ms = request.request_time_ms
    resp.request_received_time_ms = request_received_time_ms
    resp.response_time_ms = current_milli_time()
    return resp


# --------------------------------------------------------------------------- #
# Checkpoint serialization (metadata only — no raw frames)
#
# Logical recovery API ←→ implementation mapping:
#   get_state_snapshot(key)            -> to_snapshot(session_id=key)
#   get_log_tail_since(key, version)   -> get_log_tail(session_id=key,
#                                                     since_version=version)
#   apply_state_snapshot(key, snap)    -> apply_snapshot(snap)  (snap.key=key)
#   apply_log_tail(key, updates)       -> apply_log_tail(session_id=key,
#                                                       entries=updates)
# All four are exposed over gRPC as get_checkpoint/get_log_tail/
# apply_checkpoint/apply_log_tail; the router in
# edge_manager.grpc_service_application_benchmarks dispatches on snapshot.app
# or request.app.
# --------------------------------------------------------------------------- #
def _video_id_default():
    return getattr(configs, "SLAM_VIDEO_ID",
                   getattr(configs, "SLAM_VF_SESSION_ID", "default"))


def to_snapshot(session_id):
    """Build a CheckpointSnapshot for one SLAM-VF session.

    Returns ``None`` if the session does not exist (so the pusher can skip).
    """
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return None
        snap = pb2.CheckpointSnapshot()
        snap.app = "SLAM-VF"
        snap.key = session_id
        snap.version = int(session.version)
        snap.last_op_id = ""
        # Stamp the checkpoint watermark so subsequent INCREMENTAL recoveries
        # can request the log tail from this exact version.
        session.last_checkpoint_version = int(session.version)
        slam = snap.slam
        slam.video_id = _video_id_default()
        slam.state_version = int(session.version)
        slam.last_frame_id = int(session.last_frame_id)
        tx, ty, theta = _decompose_pose_xytheta(session.pose)
        slam.pose_x = float(tx)
        slam.pose_y = float(ty)
        slam.pose_theta = float(theta)
        slam.keyframe_count = int(session.num_keyframes)
        slam.map_point_count = int(session.num_map_points)
        for (frame_id, tx_v, ty_v, tz_v, theta_v, nin) in session.keyframes:
            kr = slam.keyframes.add()
            kr.frame_id = int(frame_id)
            kr.tx = float(tx_v)
            kr.ty = float(ty_v)
            kr.tz = float(tz_v)
            kr.theta = float(theta_v)
            kr.num_inliers = int(nin)
        return snap


def apply_snapshot(snap):
    """Install a CheckpointSnapshot for a SLAM-VF session into local state.

    The previous frame's grayscale/keypoints/descriptors are intentionally
    not restored — checkpoints carry metadata only — so the first frame
    after restore behaves like a fresh start (tracked=False until the next
    pair has matchable descriptors).
    """
    session_id = snap.key or snap.slam.video_id or "default"
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = _SessionState()
            _sessions[session_id] = session
        slam = snap.slam
        session.version = int(slam.state_version or snap.version or 0)
        # Receivers also adopt the checkpoint watermark so they can report it
        # back via get_state_version()-style queries and reject mis-aligned
        # incremental log tails.
        session.last_checkpoint_version = int(session.version)
        session.last_frame_id = int(slam.last_frame_id)
        session.num_keyframes = int(slam.keyframe_count)
        session.num_map_points = int(slam.map_point_count)
        # Rebuild the running pose's translation from the decomposed snapshot;
        # rotation about Y is reconstructed from pose_theta. This is a coarse
        # round-trip (we only checkpoint a compact decomposition), which is
        # acceptable for a benchmark and well-documented above.
        c, s = math.cos(slam.pose_theta), math.sin(slam.pose_theta)
        P = np.eye(4, dtype=np.float32)
        P[0, 0] = c;  P[0, 2] = s
        P[2, 0] = -s; P[2, 2] = c
        P[0, 3] = float(slam.pose_x)
        P[2, 3] = float(slam.pose_y)
        session.pose = P
        history_max = getattr(configs, "SLAM_VF_CHECKPOINT_KEYFRAME_HISTORY", 50)
        session.keyframes = deque(maxlen=max(1, int(history_max)))
        for kr in slam.keyframes:
            session.keyframes.append((
                int(kr.frame_id), float(kr.tx), float(kr.ty),
                float(kr.tz), float(kr.theta), int(kr.num_inliers),
            ))
        # Reset the log tail: post-install, the next log entries will be the
        # ones we replay during incremental recovery (or fresh frames).
        log_max = getattr(configs, "SLAM_VF_LOG_TAIL_MAX", 2000)
        session.log_tail = deque(maxlen=max(1, int(log_max)))
        # Idempotency window resets on snapshot install — pre-checkpoint req_ids
        # are no longer relevant after a restore.
        rid_max = getattr(configs, "SLAM_VF_REQ_ID_CACHE_MAX", 1000)
        session.req_history = deque(maxlen=max(1, int(rid_max)))
        # Descriptors / keypoints / prev_gray remain None on the backup.


def get_state_version(session_id):
    with _lock:
        session = _sessions.get(session_id)
        return int(session.version) if session is not None else 0


def get_log_tail(session_id, since_version):
    """Return all log-tail entries with ``version > since_version`` for a session.

    Metadata only — no raw frames, no descriptors. Used by incremental recovery
    to bring a stale replica forward from a checkpoint at ``since_version``.
    """
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return []
        return [e for e in session.log_tail if e[0] > since_version]


def apply_log_tail(session_id, entries):
    """Replay a sequence of log entries onto a session, advancing its state.

    Entries are expected in monotonically-increasing version order. Entries
    whose version is <= the current session version are skipped (idempotent).
    Returns (applied_count, last_version_after).
    """
    applied = 0
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = _SessionState()
            _sessions[session_id] = session
        for e in entries:
            ver = int(e.version)
            if ver <= session.version:
                continue
            session.version = ver
            session.last_frame_id = int(e.frame_id)
            # Pose reconstruction from the entry's (x, y, theta) — same X-Z
            # convention used in snapshots.
            c, s = math.cos(float(e.pose_theta)), math.sin(float(e.pose_theta))
            P = np.eye(4, dtype=np.float32)
            P[0, 0] = c;  P[0, 2] = s
            P[2, 0] = -s; P[2, 2] = c
            P[0, 3] = float(e.pose_x)
            P[2, 3] = float(e.pose_y)
            session.pose = P
            if e.was_keyframe:
                session.num_keyframes += 1
                session.num_map_points += int(e.num_inliers)
                session.keyframes.append((
                    int(e.frame_id),
                    float(e.pose_x), 0.0, float(e.pose_y),
                    float(e.pose_theta), int(e.num_inliers),
                ))
            session.log_tail.append((
                ver, int(e.frame_id), bool(e.tracked),
                int(e.num_features), int(e.num_matches), int(e.num_inliers),
                bool(e.was_keyframe),
                float(e.pose_x), float(e.pose_y), float(e.pose_theta),
            ))
            applied += 1
        return applied, int(session.version)


# --------------------------------------------------------------------------- #
# Primary-driven checkpoint pusher
# --------------------------------------------------------------------------- #
def _apply_checkpoint_to_peer(host, snap):
    target = f"{host}:{EDGE_GRPC_PORT}"
    try:
        with grpc.insecure_channel(target) as ch:
            stub = pb2_grpc.ApplicationBenchmarksStub(ch)
            _ = stub.apply_checkpoint(snap, timeout=5.0)
        logging.info(
            f"[SLAM-VF][Checkpoint] Applied snap v={snap.version} key={snap.key} -> {target}")
    except Exception as e:
        logging.warning(
            f"[SLAM-VF][Checkpoint] Failed to push v={snap.version} key={snap.key} -> {target}: {e}")


def _push_checkpoint_loop():
    while True:
        time.sleep(_checkpoint_period)
        if ROLE != pb2.ROLE_PRIMARY:
            continue
        with _lock:
            session_ids = list(_sessions.keys())
        for sid in session_ids:
            snap = to_snapshot(sid)
            if snap is None:
                continue
            for host in PEER_HOSTS:
                _apply_checkpoint_to_peer(host, snap)


def ensure_checkpoint_thread_running():
    global _checkpoint_thread
    with _checkpoint_thread_lock:
        if _checkpoint_thread is None or not _checkpoint_thread.is_alive():
            t = threading.Thread(target=_push_checkpoint_loop, daemon=True,
                                 name="slam-vf-checkpoint-pusher")
            t.start()
            _checkpoint_thread = t
            logging.info(
                f"[SLAM-VF][Checkpoint] pusher started (period={_checkpoint_period}s)")


def set_role_and_peers(role, peer_hosts):
    """Called by the shared set_role_and_peers RPC."""
    global ROLE, PEER_HOSTS
    ROLE = role
    PEER_HOSTS = list(peer_hosts)
    logging.info(f"[SLAM-VF][Role] role={ROLE} peers={PEER_HOSTS}")
    if ROLE == pb2.ROLE_PRIMARY:
        ensure_checkpoint_thread_running()


def set_checkpoint_period(seconds):
    global _checkpoint_period
    _checkpoint_period = int(seconds)
    logging.info(f"[SLAM-VF][Checkpoint] period set to {_checkpoint_period}s")


# --------------------------------------------------------------------------- #
# RPC adapters (called from edge_manager.grpc_service_application_benchmarks)
# --------------------------------------------------------------------------- #
def rpc_get_checkpoint(request, context):
    """Server-side get_checkpoint for SLAM-VF.

    Caller is expected to pass VersionRequest(app='SLAM-VF', key=session_id).
    For legacy EmptyProto callers, falls back to configs.SLAM_VF_SESSION_ID.
    """
    session_id = getattr(request, "key", "") or getattr(configs, "SLAM_VF_SESSION_ID", "default")
    snap = to_snapshot(session_id)
    if snap is None:
        # Empty-but-valid snapshot so the caller can still inspect app/key.
        snap = pb2.CheckpointSnapshot()
        snap.app = "SLAM-VF"
        snap.key = session_id
        snap.slam.video_id = _video_id_default()
    return snap


def rpc_apply_checkpoint(request, context):
    apply_snapshot(request)
    return pb2.Ack(ok=True, msg=f"slam-vf applied v={request.version} key={request.key}")


def rpc_get_current_version(request, context):
    session_id = getattr(request, "key", "") or getattr(configs, "SLAM_VF_SESSION_ID", "default")
    v = get_state_version(session_id)
    resp = pb2.VersionResponse(key=session_id, state_version=v)
    resp.app = "SLAM-VF"
    return resp


def rpc_get_log_tail(request, context):
    """Server-side get_log_tail for SLAM-VF.

    Returns log entries with ``version > request.since_version`` for the
    session identified by ``request.key``.
    """
    session_id = getattr(request, "key", "") or getattr(configs, "SLAM_VF_SESSION_ID", "default")
    since = int(getattr(request, "since_version", 0))
    entries = get_log_tail(session_id, since)
    lt = pb2.LogTail()
    lt.app = "SLAM-VF"
    lt.key = session_id
    lt.from_version = since
    lt.to_version = entries[-1][0] if entries else since
    for (ver, fid, tracked, nfeat, nmatch, ninl, was_kf, px, py, pth) in entries:
        e = lt.slam_entries.add()
        e.version = ver
        e.frame_id = fid
        e.tracked = tracked
        e.num_features = nfeat
        e.num_matches = nmatch
        e.num_inliers = ninl
        e.was_keyframe = was_kf
        e.pose_x = px
        e.pose_y = py
        e.pose_theta = pth
    return lt


def rpc_apply_log_tail(request, context):
    """Server-side apply_log_tail for SLAM-VF."""
    session_id = getattr(request, "key", "") or getattr(configs, "SLAM_VF_SESSION_ID", "default")
    applied, last_v = apply_log_tail(session_id, request.slam_entries)
    return pb2.Ack(ok=True,
                   msg=f"slam-vf log_tail applied={applied} version={last_v}")
