import pandas as pd
import numpy as np
import glob
import csv
import os
import logging
import configs

TIMEOUTS_FILE = configs.PROJECT_PATH + f"EDB/results_over_time/{configs.DEVICE_TYPE}_Timeouts.csv"

def _glob_fault_free_files(app, repl_mode, quorum):
    """
    Find all primary, 'normal' fault-free latency CSVs across all replicas for:
      <DEVICE>_<edgeName><N>/<repl_mode>_<quorum>/<app>-NoFault-*-normal-primary-Latency.csv
    Also accept legacy names without '-normal-primary-' to be robust.
    """
    base = configs.PROJECT_PATH + "EDB/results_over_time/"
    mode_dir = f"{repl_mode}_{quorum}"

    # new naming (with phase+role)
    pat1 = os.path.join(base, f"{configs.DEVICE_TYPE}_*", mode_dir, f"{app}-NoFault-*-normal-primary-Latency.csv")
    # legacy naming (older runs)
    pat2 = os.path.join(base, f"{configs.DEVICE_TYPE}_*", mode_dir, f"{app}-NoFault-*-Latency.csv")

    files = sorted(set(glob.glob(pat1) + glob.glob(pat2)))
    return files

def calibrate_timeouts(app, repl_mode, quorum=None):
    """
    Compute multiple timeout thresholds (tight, moderate, conservative)
    per device_type × app × repl_mode × quorum, from fault-free runs.
    Updates configs.TIMEOUT_THRESHOLDS[app] with a list of values.

    Persists results in <DEVICE>_Timeouts.csv.
    """
    quorum = quorum or ("MAJORITY" if repl_mode == "ACTIVE" else "NA")

    # Try cached thresholds first
    if os.path.exists(TIMEOUTS_FILE):
        try:
            df = pd.read_csv(TIMEOUTS_FILE)
            row = df[
                (df["device_type"] == configs.DEVICE_TYPE) &
                (df["app"] == app) &
                (df["replication_mode"] == repl_mode) &
                (df["quorum_mode"] == quorum)
            ]
            if not row.empty and "timeouts" in row.columns:
                timeouts_str = row.iloc[0]["timeouts"]
                thresholds = [int(x) for x in timeouts_str.strip("[]").split(",")]
                configs.TIMEOUT_THRESHOLDS[app] = thresholds
                logging.info(f"[CALIBRATION] Using cached thresholds for {configs.DEVICE_TYPE}-{app}-{repl_mode}-{quorum}: {thresholds}")
                return thresholds
        except Exception as e:
            logging.warning(f"[CALIBRATION] Could not read {TIMEOUTS_FILE}: {e}")

    # Collect latencies from fault-free CSVs
    files = _glob_fault_free_files(app, repl_mode, quorum)
    if not files:
        logging.warning(f"[CALIBRATION] No fault-free CSVs found for {app}-{repl_mode}-{quorum}. Did fault-free run yet?")
        return []

    latencies = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "end_to_end_latency" in df.columns:
                latencies.extend(df["end_to_end_latency"].dropna().tolist())
        except Exception as e:
            logging.warning(f"[CALIBRATION] Failed to parse {f}: {e}")

    if not latencies:
        logging.warning(f"[CALIBRATION] No latencies extracted for {app}-{repl_mode}-{quorum}.")
        return []

    arr = np.array(latencies)
    p50 = float(np.percentile(arr, 50))   # median
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    thresholds = [
        int(max(1, p50 * 1.1)),   # Tight: just above median
        int(max(1, p95 * 1.2)),   # Moderate: above p95
        # int(max(1, p99 * 2.0)),   # Conservative: 2×p99
    ]

    row = {
        "device_type": configs.DEVICE_TYPE,
        "app": app,
        "replication_mode": repl_mode,
        "quorum_mode": quorum,
        "mean": round(float(np.mean(arr)), 2),
        "median": round(p50, 2),
        "p95": round(p95, 2),
        # "p99": round(p99, 2),
        "timeouts": str(thresholds),   # store list as string
    }

    # persist
    os.makedirs(os.path.dirname(TIMEOUTS_FILE), exist_ok=True)
    file_exists = os.path.isfile(TIMEOUTS_FILE)
    with open(TIMEOUTS_FILE, "a", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # apply live
    configs.TIMEOUT_THRESHOLDS[app] = thresholds
    logging.info(f"[CALIBRATION] Saved & applied thresholds for {configs.DEVICE_TYPE}-{app}-{repl_mode}-{quorum}: {thresholds}")
    return thresholds
