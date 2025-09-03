import threading
import json
import os
import uuid
from collections import deque
from textblob import TextBlob
from utils import current_milli_time
from protos import benchmark_pb2 as pb2

CHECKPOINT_FILE = "sentiment_state.json"
WINDOW_SIZE = 50  # keep last 50 updates per key


class SentimentAggregator:
    def __init__(self):
        self.lock = threading.RLock()
        self.state = {}         # key -> {"window": deque, "polarity": float, "subjectivity": float}
        self.version = 0
        self.last_op_id = ""
        self.seen_req_ids = set()

    def _score_text(self, text: str):
        blob = TextBlob(text)
        polarity, subjectivity = 0, 0
        for s in blob.sentences:
            polarity += s.sentiment.polarity
            subjectivity += s.sentiment.subjectivity
        total = len(blob.sentences) if len(blob.sentences) > 0 else 1
        return polarity / total, subjectivity / total

    def update(self, key: str, text: str, req_id: str = None):
        if req_id is None:
            req_id = str(uuid.uuid4())

        with self.lock:
            if req_id in self.seen_req_ids:
                # duplicate request, ignore
                agg = self.state.get(key, {"polarity": 0, "subjectivity": 0})
                return agg["polarity"], agg["subjectivity"], self.version, False

            polarity, subjectivity = self._score_text(text)
            if key not in self.state:
                self.state[key] = {
                    "window": deque(maxlen=WINDOW_SIZE),
                    "polarity": 0.0,
                    "subjectivity": 0.0
                }

            self.state[key]["window"].append((polarity, subjectivity))
            # recompute averages
            w = self.state[key]["window"]
            avg_p = sum(p for p, _ in w) / len(w)
            avg_s = sum(s for _, s in w) / len(w)
            self.state[key]["polarity"] = avg_p
            self.state[key]["subjectivity"] = avg_s

            # update global version
            self.version += 1
            self.last_op_id = req_id
            self.seen_req_ids.add(req_id)

            return avg_p, avg_s, self.version, True

    def checkpoint(self):
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

    def restore(self):
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


# Singleton instance
aggregator = SentimentAggregator()
aggregator.restore()


def analyze_sentiment_stateful(request, request_received_time_ms):
    polarity, subjectivity, version, applied = aggregator.update(
        key=request.key,
        text=request.input_text,
        req_id=request.req_id
    )

    response = pb2.SentimentAggregationResponse()
    response.key = request.key
    response.req_id = request.req_id
    response.polarity = polarity
    response.subjectivity = subjectivity
    response.state_version = version
    response.applied = applied
    response.request_time_ms = request.request_time_ms
    response.request_received_time_ms = request_received_time_ms
    response.response_time_ms = current_milli_time()

    return response
