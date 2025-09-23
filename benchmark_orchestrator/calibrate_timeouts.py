import os
import glob
import pandas as pd
import numpy as np
import json

RESULTS_BASE = os.path.join('..', 'results_over_time')  # adjust to your path
OUT_FILE = "calibrated_timeouts.json"

# Percentiles to use per mode
PERCENTILES = {
    "BASELINE": 95,
    "PASSIVE": 95,
    "ACTIVE": 99,
}

def parse_latency_file(filepath):
    """Read one latency CSV and return end-to-end latencies."""
    try:
        df = pd.read_csv(filepath)
        return df["end_to_end_latency"].dropna().astype(float).tolist()
    except Exception as e:
        print(f"[WARN] Could not parse {filepath}: {e}")
        return []

def collect_latencies(device_dir, repl_mode, quorum, app):
    """Collect all latencies across rounds for one (device, repl, quorum, app)."""
    pattern = os.path.join(
        device_dir,
        f"{repl_mode}_{quorum}",
        f"{app}-NoFault-*.csv"
    )
    files = glob.glob(pattern)
    latencies = []
    for f in files:
        latencies.extend(parse_latency_file(f))
    return latencies

def calibrate_timeouts(results_base=RESULTS_BASE, percentiles=PERCENTILES):
    timeouts = {}
    device_dirs = glob.glob(os.path.join(results_base, "*"))

    for device_dir in device_dirs:
        device_type = os.path.basename(device_dir).split("_")[0]  # "pi" or "nano"
        timeouts.setdefault(device_type, {})

        for repl_mode, perc in percentiles.items():
            quorum_modes = ["NA"] if repl_mode in ("BASELINE", "PASSIVE") else ["MAJORITY"]
            timeouts[device_type].setdefault(repl_mode, {})

            for quorum in quorum_modes:
                timeouts[device_type][repl_mode].setdefault(quorum, {})

                # Scan apps
                app_dirs = glob.glob(os.path.join(device_dir, f"{repl_mode}_{quorum}", "*.csv"))
                apps = sorted({os.path.basename(f).split("-")[0] for f in app_dirs})

                for app in apps:
                    latencies = collect_latencies(device_dir, repl_mode, quorum, app)
                    if not latencies:
                        continue
                    thr = int(np.percentile(latencies, perc))
                    timeouts[device_type][repl_mode][quorum][app] = thr
                    print(f"[{device_type}] {repl_mode}_{quorum} {app}: P{perc}={thr}ms ({len(latencies)} samples)")

    # Save to JSON
    with open(OUT_FILE, "w") as f:
        json.dump(timeouts, f, indent=2)
    print(f"[DONE] Calibrated thresholds saved to {OUT_FILE}")
    return timeouts

if __name__ == "__main__":
    calibrate_timeouts()
