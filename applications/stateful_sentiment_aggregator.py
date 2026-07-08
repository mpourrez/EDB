# sa_agg.py (or your SA-AGG handler module)

import threading
import json
import os
import uuid
import logging
import time
from collections import deque

import grpc
from textblob import TextBlob
from utils import current_milli_time
from protos import benchmark_pb2 as pb2
from protos import benchmark_pb2_grpc as pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

CHECKPOINT_FILE = "sentiment_state.json"
WINDOW_SIZE = 50
MAX_REQ_IDS = 5000
# Bounded operation-log tail retained for INCREMENTAL recovery. Each entry is
# a compact, already-scored update (no raw input text). Sized so it comfortably
# covers the ops accumulated between two checkpoint watermarks.
MAX_LOG_TAIL = 5000

# ===== Runtime config set via RPCs =====
ROLE = pb2.ROLE_BACKUP           # default
PEER_HOSTS = []                  # other replicas (strings)
EDGE_GRPC_PORT = int(os.getenv("EDGE_GRPC_PORT", "50051"))

_checkpoint_period = 30          # seconds
_checkpoint_thread = None
_checkpoint_lock = threading.Lock()

class SentimentAggregator:
    def __init__(self):
        self.lock = threading.RLock()
        self.state = {}        # key -> {window, polarity, subjectivity}
        # ``version`` is the current_state_version (monotonic update counter).
        # ``last_checkpoint_version`` is the version captured by the most
        # recent to_snapshot() call. Together they let the orchestrator
        # decide between FULL and INCREMENTAL recovery transfers.
        self.version = 0
        self.last_checkpoint_version = 0
        self.last_op_id = ""
        self.req_history = {}  # key -> deque of last MAX_REQ_IDS
        # Bounded operation-log tail for INCREMENTAL recovery.
        # Each entry: (version, key, req_id, polarity, subjectivity) where
        # (polarity, subjectivity) are the per-update TextBlob scores — never
        # the raw input text — so a stale replica can replay them cheaply.
        self.log_tail = deque(maxlen=MAX_LOG_TAIL)

    def _score_text(self, text: str):
        blob = TextBlob(text)
        polarity, subjectivity = 0.0, 0.0
        total = max(1, len(blob.sentences))
        for s in blob.sentences:
            polarity += s.sentiment.polarity
            subjectivity += s.sentiment.subjectivity
        return polarity / total, subjectivity / total

    def update(self, key: str, text: str, req_id: str = None):
        if req_id is None:
            req_id = str(uuid.uuid4())
        with self.lock:
            if key not in self.req_history:
                self.req_history[key] = deque(maxlen=MAX_REQ_IDS)
            if req_id in self.req_history[key]:
                agg = self.state.get(key, {"polarity": 0.0, "subjectivity": 0.0})
                return agg["polarity"], agg["subjectivity"], self.version, False
            self.req_history[key].append(req_id)

            p, s = self._score_text(text)
            if key not in self.state:
                self.state[key] = {
                    "window": deque(maxlen=WINDOW_SIZE),
                    "polarity": 0.0,
                    "subjectivity": 0.0
                }
            self.state[key]["window"].append((p, s))
            w = self.state[key]["window"]
            self.state[key]["polarity"] = sum(pp for pp, _ in w) / len(w)
            self.state[key]["subjectivity"] = sum(ss for _, ss in w) / len(w)

            self.version += 1
            self.last_op_id = req_id
            # Record the applied op in the log tail (only on applied=True, i.e.
            # a genuinely new req_id) so it can be replayed after a checkpoint.
            # Store the already-computed (p, s) — not the raw text.
            self.log_tail.append((self.version, key, req_id, p, s))
            return self.state[key]["polarity"], self.state[key]["subjectivity"], self.version, True

    def _apply_scored_update(self, key, req_id, polarity, subjectivity, version):
        """Replay one already-scored update onto local state (caller holds lock).

        Mirrors the state math in update() but consumes pre-computed
        (polarity, subjectivity) instead of re-running TextBlob, so recovery
        never needs the raw input text. Returns True if applied, False if the
        req_id was already seen for this key (idempotent skip). Advances
        self.version to ``version`` on apply and appends to the local log tail
        so a recovered replica can itself serve get_log_tail().
        """
        if key not in self.req_history:
            self.req_history[key] = deque(maxlen=MAX_REQ_IDS)
        if req_id and req_id in self.req_history[key]:
            return False
        if req_id:
            self.req_history[key].append(req_id)
        if key not in self.state:
            self.state[key] = {
                "window": deque(maxlen=WINDOW_SIZE),
                "polarity": 0.0,
                "subjectivity": 0.0
            }
        w = self.state[key]["window"]
        w.append((polarity, subjectivity))
        self.state[key]["polarity"] = sum(pp for pp, _ in w) / len(w)
        self.state[key]["subjectivity"] = sum(ss for _, ss in w) / len(w)
        self.version = int(version)
        self.last_op_id = req_id
        self.log_tail.append((int(version), key, req_id,
                              float(polarity), float(subjectivity)))
        return True

    def get_log_tail(self, since_version):
        """Return log entries with version > since_version, oldest first.

        Returns ``(entries, oldest_retained, current_version)`` where
        ``oldest_retained`` is the version of the oldest entry still held in
        the bounded log (or None if the log is empty). The caller uses these
        to detect a gap: if ``since_version`` predates ``oldest_retained`` some
        ops have been evicted and the caller must fall back to FULL recovery.
        """
        with self.lock:
            since = int(since_version)
            entries = [e for e in self.log_tail if e[0] > since]
            oldest = int(self.log_tail[0][0]) if self.log_tail else None
            return entries, oldest, int(self.version)

    def apply_log_tail(self, entries):
        """Replay scored update entries onto local state, idempotently.

        ``entries`` is an iterable of pb2.SAAGGLogEntry (fields version/key/
        req_id/polarity/subjectivity). Entries are sorted by version before
        applying; those with version <= current self.version are skipped, and
        duplicate req_ids are skipped. Never decreases self.version. Returns
        ``(applied, skipped, final_version)``.
        """
        applied = 0
        skipped = 0
        with self.lock:
            ordered = sorted(entries, key=lambda e: int(e.version))
            for e in ordered:
                ver = int(e.version)
                if ver <= self.version:
                    # Already reflected in local state (or a re-applied tail).
                    skipped += 1
                    continue
                if ver > self.version + 1:
                    logging.warning(
                        f"SA_LOG_TAIL_GAP key={e.key} expected={self.version + 1} "
                        f"got={ver} (replaying anyway, state may drift)"
                    )
                ok = self._apply_scored_update(
                    e.key, e.req_id, float(e.polarity),
                    float(e.subjectivity), ver
                )
                if ok:
                    applied += 1
                else:
                    skipped += 1
            return applied, skipped, int(self.version)

    def to_snapshot(self, key=""):
        """Return an in-memory checkpoint snapshot (no file I/O).

        Logical-API alias: ``get_state_snapshot(key)`` — exposed over gRPC as
        ``ApplicationBenchmarks.get_checkpoint(VersionRequest{app, key})``.
        """
        with self.lock:
            snap = pb2.CheckpointSnapshot()
            snap.app = "SA-AGG"
            snap.key = key or ""
            snap.version = self.version
            snap.last_op_id = self.last_op_id
            # Stamp the checkpoint watermark so an incremental recovery caller
            # can decide whether a log-tail transfer is needed (SA-AGG has no
            # log model, so the orchestrator collapses to a full transfer).
            self.last_checkpoint_version = int(self.version)
            for k, v in self.state.items():
                entry = pb2.CheckpointSnapshot.Entry(
                    polarity=v["polarity"],
                    subjectivity=v["subjectivity"]
                )
                # window → two arrays (polarity, subjectivity)
                for p, s in v["window"]:
                    entry.window_p.append(p)
                    entry.window_s.append(s)
                snap.state[k].CopyFrom(entry)
            return snap

    def apply_snapshot(self, snap: pb2.CheckpointSnapshot):
        """Install a snapshot into local state.

        Logical-API alias: ``apply_state_snapshot(key, snap)`` — exposed over
        gRPC as ``ApplicationBenchmarks.apply_checkpoint(CheckpointSnapshot)``.
        """
        with self.lock:
            self.version = snap.version
            self.last_checkpoint_version = int(snap.version)
            self.last_op_id = snap.last_op_id
            self.state = {}
            # Reset idempotency + log tail: pre-checkpoint req_ids and log
            # entries are stale after installing a snapshot. The version
            # watermark (self.version) prevents any pre-snapshot op from being
            # replayed, and the next entries appended will be the ones we
            # replay during incremental recovery (or fresh updates).
            self.req_history = {}
            self.log_tail = deque(maxlen=MAX_LOG_TAIL)
            for k, entry in snap.state.items():
                window = deque(maxlen=WINDOW_SIZE)
                # Rebuild window pairs
                for i in range(min(len(entry.window_p), len(entry.window_s))):
                    window.append((entry.window_p[i], entry.window_s[i]))
                self.state[k] = {
                    "window": window,
                    "polarity": entry.polarity,
                    "subjectivity": entry.subjectivity
                }

    # (optional) Keep disk checkpoint for crash restart
    def checkpoint_to_disk(self):
        with self.lock:
            snapshot = {
                "version": self.version,
                "last_op_id": self.last_op_id,
                "state": {
                    k: {
                        "polarity": v["polarity"],
                        "subjectivity": v["subjectivity"],
                        "window": list(v["window"])
                    } for k, v in self.state.items()
                }
            }
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(snapshot, f)
            return snapshot

    def restore_from_disk(self):
        if not os.path.exists(CHECKPOINT_FILE):
            return
        with open(CHECKPOINT_FILE, "r") as f:
            snapshot = json.load(f)
        with self.lock:
            self.version = snapshot.get("version", 0)
            self.last_op_id = snapshot.get("last_op_id", "")
            self.state = {}
            for k, v in snapshot["state"].items():
                dq = deque(v["window"], maxlen=WINDOW_SIZE)
                self.state[k] = {
                    "polarity": v["polarity"],
                    "subjectivity": v["subjectivity"],
                    "window": dq
                }
            if not hasattr(self, "req_history"):
                self.req_history = {}

aggregator = SentimentAggregator()
aggregator.restore_from_disk()

# ===== Primary-driven checkpoint pusher =====

def _apply_checkpoint_to_peer(host: str, snap: pb2.CheckpointSnapshot):
    target = f"{host}:{EDGE_GRPC_PORT}"
    try:
        with grpc.insecure_channel(target) as ch:
            stub = pb2_grpc.ApplicationBenchmarksStub(ch)
            _ = stub.apply_checkpoint(snap, timeout=5.0)
        logging.info(f"[Checkpoint] Applied snapshot v={snap.version} to {target}")
    except Exception as e:
        logging.warning(f"[Checkpoint] Failed to apply snapshot v={snap.version} to {target}: {e}")

def _push_checkpoint_loop():
    global _checkpoint_period
    while True:
        time.sleep(_checkpoint_period)
        # Primary only
        if ROLE != pb2.ROLE_PRIMARY:
            continue
        snap = aggregator.to_snapshot(key="")
        # (optional) also persist locally for crash restart
        aggregator.checkpoint_to_disk()
        # Push to all peers
        for host in PEER_HOSTS:
            _apply_checkpoint_to_peer(host, snap)

def _ensure_checkpoint_thread_running():
    global _checkpoint_thread
    with _checkpoint_lock:
        if _checkpoint_thread is None or not _checkpoint_thread.is_alive():
            t = threading.Thread(target=_push_checkpoint_loop, daemon=True, name="checkpoint-pusher")
            t.start()
            _checkpoint_thread = t
            logging.info(f"[Checkpoint] pusher started (period={_checkpoint_period}s)")

# ===== RPC handlers to wire into your gRPC server =====

def rpc_set_role_and_peers(request, context):
    """EdgeService.SetRoleAndPeers"""
    global ROLE, PEER_HOSTS
    ROLE = request.role
    PEER_HOSTS = list(request.peer_hosts)
    logging.info(f"[Role] Set role={ROLE} peers={PEER_HOSTS}")
    if ROLE == pb2.ROLE_PRIMARY:
        _ensure_checkpoint_thread_running()
    return pb2.Ack(ok=True, msg="role/peers updated")

def rpc_set_checkpoint_period(request, context):
    """EdgeService.SetCheckpointPeriod"""
    global _checkpoint_period
    _checkpoint_period = int(request.seconds)
    logging.info(f"[Checkpoint] period set to {_checkpoint_period}s")
    return pb2.Ack(ok=True, msg="period updated")

def rpc_get_checkpoint(request, context):
    """EdgeService.GetCheckpoint"""
    return aggregator.to_snapshot(key=getattr(request, "key", ""))

def rpc_apply_checkpoint(request, context):
    """EdgeService.ApplyCheckpoint (backup side)"""
    aggregator.apply_snapshot(request)
    # (optional) keep a disk copy on backup too
    aggregator.checkpoint_to_disk()
    return pb2.Ack(ok=True, msg=f"applied v={request.version}")

def analyze_sentiment_stateful(request, request_received_time_ms):
    p, s, version, applied = aggregator.update(
        key=request.key, text=request.input_text, req_id=request.req_id
    )
    resp = pb2.SentimentAggregationResponse()
    resp.key = request.key
    resp.req_id = request.req_id
    resp.polarity = p
    resp.subjectivity = s
    resp.state_version = version
    resp.applied = applied
    resp.request_time_ms = request.request_time_ms
    resp.request_received_time_ms = request_received_time_ms
    resp.response_time_ms = current_milli_time()
    return resp

def get_current_version(request, context):
    resp = pb2.VersionResponse(key=request.key, state_version=aggregator.version)
    resp.app = "SA-AGG"
    return resp


def rpc_get_log_tail(request, context):
    """Server-side get_log_tail for SA-AGG.

    Returns the operation-log tail with ``version > request.since_version`` as
    pb2.SAAGGLogEntry records. If the requested watermark predates the oldest
    entry still retained in the bounded log, some ops have been evicted and the
    caller must fall back to FULL recovery (signalled via SA_LOG_TAIL_MISS).
    """
    key = getattr(request, "key", "") or ""
    since = int(getattr(request, "since_version", 0))
    entries, oldest, current = aggregator.get_log_tail(since)

    lt = pb2.LogTail()
    lt.app = "SA-AGG"
    lt.key = key
    lt.from_version = since
    lt.to_version = entries[-1][0] if entries else since
    for (ver, k, rid, pol, subj) in entries:
        e = lt.sa_entries.add()
        e.version = int(ver)
        e.key = k
        e.req_id = rid
        e.polarity = float(pol)
        e.subjectivity = float(subj)

    # Gap detection: we can only serve ops we still retain. A gap exists when
    # the oldest retained entry is newer than since+1, or the log is empty yet
    # our version is ahead of `since` (nothing left to replay the delta).
    miss = ((oldest is not None and oldest > since + 1) or
            (oldest is None and current > since))
    if miss:
        logging.warning(
            f"SA_LOG_TAIL_MISS since={since} oldest_retained={oldest} "
            f"current_version={current} count={len(entries)} "
            f"-> caller should fall back to FULL recovery"
        )
    else:
        logging.info(
            f"SA_LOG_TAIL_GET since={since} count={len(entries)} "
            f"from_version={lt.from_version} to_version={lt.to_version}"
        )
    return lt


def rpc_apply_log_tail(request, context):
    """Server-side apply_log_tail for SA-AGG (backup / recovering side)."""
    applied, skipped, final_v = aggregator.apply_log_tail(request.sa_entries)
    logging.info(
        f"SA_LOG_TAIL_APPLY applied={applied} skipped={skipped} "
        f"final_version={final_v}"
    )
    return pb2.Ack(ok=True,
                   msg=f"sa-agg log_tail applied={applied} skipped={skipped} "
                       f"version={final_v}")
