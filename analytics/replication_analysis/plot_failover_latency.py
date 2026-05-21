#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, glob, os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
EVENT_FILES = {
    "Raspberry Pi": "pi_Replication-Events.csv",
    "Jetson Nano":  "nano_Replication-Events.csv",
}
ROOT_LAT = "../../results_over_time"  # folder with latency csvs
APP = "SA-AGG"
OUT = "failover_latency_bar.png"

# -------------------------------------------------
# 1. Load and find failover windows
# -------------------------------------------------
def load_events():
    frames = []
    for dev, f in EVENT_FILES.items():
        df = pd.read_csv(f)
        df["device"] = dev
        frames.append(df)
    ev = pd.concat(frames, ignore_index=True)
    ev["event"] = ev["event"].str.upper()
    return ev

events = load_events()

# Keep only failure-recovery runs
fail_runs = events[events["event"].isin(["CRASH_TRIGGERED","FAILOVER","PROMOTE_BACKUP","RECOVERY_FAILED"])]
# Use FAULT_INJECT_START to group runs
runs = events[events["event"]=="FAULT_INJECT_START"].copy()

# Some files have timeout threshold encoded in details or column
def extract_timeout(detail):
    m = re.search(r"Timeout:?(\d+)", str(detail))
    return int(m.group(1)) if m else np.nan

runs["timeout"] = runs["details"].apply(extract_timeout)
# Build a run id for join
runs["run_id"] = runs.groupby(["device","replication_mode","quorum_mode","fault","timeout"]).cumcount()

# -------------------------------------------------
# 2. Measure failover latency per run
# -------------------------------------------------
records = []

for _, run in runs.iterrows():
    dev = run["device"]
    repl = run["replication_mode"]
    tmo = run["timeout"]
    # group key
    key = (dev, repl, tmo)

    # all events for this (device, repl, timeout) run
    mask = (events["device"]==dev)&(events["replication_mode"]==repl)
    if pd.notna(tmo):
        mask &= events["details"].astype(str).str.contains(f"Timeout:{tmo}")
    run_events = events[mask].sort_values("timestamp_ms")

    crash = run_events[run_events["event"]=="CRASH_TRIGGERED"]["timestamp_ms"].min()
    failover = run_events[run_events["event"]=="NEW_LEADER_DETECTED"]["timestamp_ms"].min()
    if pd.isna(crash) or pd.isna(failover):  # skip incomplete runs
        continue
    failover_dur = (failover - crash) / 1000.0
    records.append({"device":dev,"replication_mode":repl,"timeout_ms":tmo,"failover_latency_s":failover_dur})

lat = pd.DataFrame(records)
if lat.empty:
    raise RuntimeError("No failover events found; check event logs")

# -------------------------------------------------
# 3. Aggregate and plot
# -------------------------------------------------
lat["replication_mode"] = pd.Categorical(lat["replication_mode"], categories=["PASSIVE","ACTIVE"], ordered=True)

agg = lat.groupby(["device","replication_mode","timeout_ms"]).agg(
    mean_lat=("failover_latency_s","mean"),
    std_lat=("failover_latency_s","std"),
    n=("failover_latency_s","count")
).reset_index()

# Plot
sns.set(style="whitegrid", font_scale=1.3)
g = sns.catplot(
    data=agg, kind="bar",
    x="timeout_ms", y="mean_lat",
    hue="replication_mode",
    col="device",
    ci=None, height=5, aspect=1.2
)
for ax, (_, sub) in zip(g.axes.flat, agg.groupby("device")):
    # add error bars manually
    dev = sub["device"].iloc[0]
    for i, row in sub.iterrows():
        ax.errorbar(
            x=row.name, y=row.mean_lat,
            yerr=row.std_lat, fmt='none', c='black', capsize=5
        )

g.set_axis_labels("Timeout (ms)", "Failover Latency (s)")
g.set_titles("{col_name}")
plt.suptitle(f"Failover Latency (mean ± SD) for {APP}", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig(OUT, dpi=300)
print(f"[x] Saved {OUT}")
