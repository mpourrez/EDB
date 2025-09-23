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
        self.version = 0
        self.last_op_id = ""
        self.req_history = {}  # key -> deque of last MAX_REQ_IDS

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
            return self.state[key]["polarity"], self.state[key]["subjectivity"], self.version, True

    def to_snapshot(self):
        """Return an in-memory checkpoint snapshot (no file I/O)."""
        with self.lock:
            snap = pb2.CheckpointSnapshot()
            snap.version = self.version
            snap.last_op_id = self.last_op_id
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
        """Install a snapshot into local state."""
        with self.lock:
            self.version = snap.version
            self.last_op_id = snap.last_op_id
            self.state = {}
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
            stub = pb2_grpc.EdgeServiceStub(ch)
            _ = stub.ApplyCheckpoint(snap, timeout=5.0)
        logging.info(f"[Checkpoint] Applied to backup {target} (v={snap.version})")
    except Exception as e:
        logging.warning(f"[Checkpoint] Failed to apply to {target}: {e}")

def _push_checkpoint_loop():
    global _checkpoint_period
    while True:
        time.sleep(_checkpoint_period)
        # Primary only
        if ROLE != pb2.ROLE_PRIMARY:
            continue
        snap = aggregator.to_snapshot()
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
    return aggregator.to_snapshot()

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
    return pb2.VersionResponse(key=request.key, state_version=aggregator.version)
