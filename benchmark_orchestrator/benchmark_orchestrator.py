import sys

sys.path.append('..')
import csv

import configs
import utils
import grpc_client
import time
import analyze_results
import multiprocessing
import benchmark_orchestrator
import os
import signal
import uuid
import logging
import calibrate_timeouts
from protos import benchmark_pb2 as pb2

# set up logging at the top of the file if not already
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
import csv
import os

def run_tag(app, fault_tag):
    """
    Build a consistent tag for filenames:
    <APP>-<REPL>-<QUORUM>-<FAULTTAG>
    QUORUM only appears for ACTIVE.
    """
    repl = getattr(configs, "REPLICATION_MODE", "BASELINE")
    quorum = getattr(configs, "QUORUM_MODE", "NA")
    parts = [app, repl]
    if repl == "ACTIVE":
        parts.append(quorum)
    parts.append(fault_tag)
    return "-".join(parts)

def get_events_file():
    return configs.PROJECT_PATH + f"EDB/results_over_time/{configs.DEVICE_TYPE}_Replication-Events.csv"

def log_replication_event(event, details=""):
    """
    Append a replication event to a CSV file.
    Example: log_replication_event("FAILOVER", "from=0,to=1")
    """
    events_file = get_events_file()
    os.makedirs(os.path.dirname(events_file), exist_ok=True)
    file_exists = os.path.isfile(events_file)

    with open(events_file, "a", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_ms", "device_type", "event", "details", "replication_mode", "quorum_mode", "app", "timeout", "fault"])

        writer.writerow([
            utils.current_milli_time(),
            configs.DEVICE_TYPE,
            event,
            details,
            configs.REPLICATION_MODE,
            getattr(configs, "QUORUM_MODE", "NA"),
            configs.CURRENT_APP,
            configs.FAIL_DETECT_TIMEOUT_MS,
            configs.CURRENT_FAULT
        ])


PROGRESS_FILE = configs.PROJECT_PATH + f"EDB/results_over_time/{configs.DEVICE_TYPE}_Experiment-Progress.csv"

def log_progress(device_type, app, repl_mode, quorum, fault, fault_config,
                 round_id, status, replica_id=None, replica_host=None,
                 phase="general", timeout_threshold_ms="NA"):
    """
    Append experiment progress to a CSV.
    status: "STARTED" | "DONE" | "FAILED:<error>"
    phase: "fault_free" | "stress" | "failure_recovery" | "checkpoint"
    timeout_threshold_ms: which timeout (for multi-threshold sweeps)
    """
    progress_file = configs.PROJECT_PATH + f"EDB/results_over_time/{device_type}_Experiment-Progress.csv"
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    file_exists = os.path.isfile(progress_file)

    with open(progress_file, "a", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_ms", "device_type", "replica_id", "replica_host",
                "app", "replication_mode", "quorum_mode",
                "fault", "fault_config", "round_id", "status", "phase",
                "timeout_threshold_ms"
            ])
        writer.writerow([
            utils.current_milli_time(),
            device_type,
            replica_id if replica_id is not None else "NA",
            replica_host if replica_host is not None else "NA",
            app,
            repl_mode,
            quorum,
            getattr(fault, "abbreviation", "NONE") if fault else "NONE",
            str(fault_config) if fault_config else "NONE",
            round_id,
            status,
            phase,
            str(timeout_threshold_ms)
        ])




import pandas as pd

def log_progress(device_type, app, repl_mode, quorum, fault, fault_config,
                 round_id, status, replica_id=None, replica_host=None,
                 phase="general", timeout_threshold_ms="NA"):
    """
    Append experiment progress to a CSV.
    status: "STARTED" | "DONE" | "FAILED:<error>"
    phase: "fault_free" | "stress" | "failure_recovery" | "checkpoint"
    timeout_threshold_ms: which timeout (for multi-threshold sweeps)
    """
    progress_file = configs.PROJECT_PATH + f"EDB/results_over_time/{device_type}_Experiment-Progress.csv"
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    file_exists = os.path.isfile(progress_file)

    with open(progress_file, "a", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_ms", "device_type", "replica_id", "replica_host",
                "app", "replication_mode", "quorum_mode",
                "fault", "fault_config", "round_id", "status", "phase",
                "timeout_threshold_ms"
            ])
        writer.writerow([
            utils.current_milli_time(),
            device_type,
            replica_id if replica_id is not None else "NA",
            replica_host if replica_host is not None else "NA",
            app,
            repl_mode,
            quorum,
            getattr(fault, "abbreviation", "NONE") if fault else "NONE",
            str(fault_config) if fault_config else "NONE",
            round_id,
            status,
            phase,
            str(timeout_threshold_ms)
        ])


def already_done(device_type, app, repl_mode, quorum, fault, fault_config,
                 round_id, replica_id, phase="general", timeout_threshold_ms="NA"):
    """
    Return True if the same experiment (including timeout threshold) is already DONE.
    """
    progress_file = configs.PROJECT_PATH + f"EDB/results_over_time/{device_type}_Experiment-Progress.csv"
    if not os.path.exists(progress_file):
        return False

    try:
        df = pd.read_csv(progress_file)
    except Exception:
        return False

    # Normalize
    for col in ["replica_id", "round_id", "fault_config", "timeout_threshold_ms"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["quorum_mode"] = df["quorum_mode"].fillna("NA")
    if "phase" not in df.columns:
        df["phase"] = "general"

    fault_abbrev = getattr(fault, "abbreviation", "NONE") if fault else "NONE"
    fault_cfg_str = str(fault_config) if fault_config else "NONE"

    subset = df[
        (df["device_type"] == str(device_type)) &
        (df["app"] == str(app)) &
        (df["replication_mode"] == str(repl_mode)) &
        (df["quorum_mode"] == str(quorum)) &
        (df["fault"] == fault_abbrev) &
        (df["fault_config"] == fault_cfg_str) &
        (df["round_id"] == str(round_id)) &
        (df["replica_id"] == str(replica_id)) &
        (df["phase"] == str(phase)) &
        (df["timeout_threshold_ms"] == str(timeout_threshold_ms))
    ]

    if subset.empty:
        return False

    last_status = subset.sort_values("timestamp_ms").iloc[-1]["status"]
    return last_status == "DONE"




import concurrent.futures
import threading
from protos import benchmark_pb2 as pb2
from grpc_client import Client

# -------------------------
# Replication primitives
# -------------------------

class ReplicaManager:
    def __init__(self, device_group):
        """device_group = PI_GROUP or NANO_GROUP"""
        self.replicas = [Client(ip) for ip, _ in device_group]
        self.primary_idx = 0
        self.last_hb = [0] * len(self.replicas)
        self.alive = [True] * len(self.replicas)

    def heartbeat_loop(self):
        while True:
            for i, c in enumerate(self.replicas):
                try:
                    c.call_ping()
                    self.last_hb[i] = utils.current_milli_time()
                except Exception:
                    # don’t mark dead here
                    logging.debug(f"[Heartbeat] Replica {i} ping failed, waiting for timeout check")
            self.detect_failures()
            time.sleep(configs.HEARTBEAT_MS / 1000.0)

    def detect_failures(self):
        now = utils.current_milli_time()
        for i, t in enumerate(self.last_hb):
            if now - t > configs.FAIL_DETECT_TIMEOUT_MS:
                if self.alive[i]:
                    # Transition: alive → dead
                    self.alive[i] = False
                    logging.warning(f"Replica {i} failed heartbeat")
                    log_replication_event("HEARTBEAT_TIMEOUT", f"replica={i}")
                    if i == self.primary_idx:
                        promote_backup(self)
            else:
                if not self.alive[i]:
                    # Transition: dead → alive (recovered)
                    self.alive[i] = True
                    logging.info(f"Replica {i} recovered heartbeat")
                    log_replication_event("HEARTBEAT_RECOVERED", f"replica={i}")

def promote_backup(rm):
    old_primary = rm.primary_idx
    for i, alive in enumerate(rm.alive):
        if i != old_primary and alive:
            rm.primary_idx = i
            logging.warning(f"Promoted replica {i} as new primary (old={old_primary})")
            log_replication_event("FAILOVER", f"from={old_primary},to={rm.primary_idx}")
            # Reconfigure roles only if mode=PASSIVE and app=SA-AGG (stateful)
            if getattr(configs, "REPLICATION_MODE", "BASELINE") == "PASSIVE":
                if getattr(configs, "CURRENT_APP", None) == "SA-AGG":
                    try:
                        _reconfigure_roles_for_passive(rm)
                    except Exception as e:
                        logging.warning(f"[role] reconfigure after failover failed: {e}")
            return
    logging.error("No alive backup available to promote!")


def _reconfigure_roles_for_passive(rm):
    primary = rm.primary_idx
    all_hosts = [c.host for c in rm.replicas]

    for i, cli in enumerate(rm.replicas):
        if i == rm.primary_idx:
            peers = [h for j, h in enumerate(all_hosts) if j != i]
            try:
                cli.call_set_role_and_peers(pb2.ROLE_PRIMARY, peers)
                logging.info(f"[role] Set role for primary on {cli.host}")
            except Exception as e:
                logging.warning(f"[role] cannot set PRIMARY on {cli.host}: {e}")
        else:
            peers = [h for j, h in enumerate(all_hosts) if j != i]
            try:
                cli.call_set_role_and_peers(pb2.ROLE_BACKUP, peers)
                logging.info(f"[role] Set role for backup on {cli.host}")
            except Exception as e:
                logging.warning(f"[role] cannot set BACKUP on {cli.host}: {e}")

def active_replicate_request(rm, request_fn, *args, **kwargs):
    """Broadcast request to all replicas, return first or quorum result."""
    results = []
    quorum_size = len(rm.replicas) // 2 + 1
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futs = [pool.submit(request_fn, c, *args, **kwargs) for c in rm.replicas]
        for f in concurrent.futures.as_completed(futs):
            try:
                res = f.result()
                if res:
                    results.append(res)
                if configs.QUORUM_MODE == "FIRST" and res:
                    return res
                if configs.QUORUM_MODE == "MAJORITY" and len(results) >= quorum_size:
                    return results[0]
            except Exception:
                continue
    return None


def passive_replicate_request(rm, request_fn, *args, **kwargs):
    """Send request to primary replica only."""
    primary = rm.replicas[rm.primary_idx]
    return request_fn(primary, *args, **kwargs)


def replicate_request(rm, app_name, request_fn, *args, **kwargs):
    """Generic replication wrapper."""
    if configs.REPLICATION_MODE == "ACTIVE":
        return active_replicate_request(rm, request_fn, *args, **kwargs)
    elif configs.REPLICATION_MODE == "PASSIVE":
        return passive_replicate_request(rm, request_fn, *args, **kwargs)
    else:
        return request_fn(rm.replicas[rm.primary_idx], *args, **kwargs)



####################################################################################
####################################################################################
## Saving latency and resource utilization results in csv file
####################################################################################
####################################################################################
def save_experiment_results_over_time(device_num, application, fault_config_file_name, results, role="primary", phase="normal"):
    """
    Save per-request latency results.
    role = "primary" or "backup"
    phase = "precrash" | "postcrash" | "normal"
    """
    base_dir = (
        configs.PROJECT_PATH
        + "EDB/results_over_time/"
        + f"{configs.DEVICE_TYPE}_{configs.EDGE_DEVICE_NAME.value}{device_num}/"
        + f"{configs.REPLICATION_MODE}_{configs.QUORUM_MODE}/"
    )
    os.makedirs(base_dir, exist_ok=True)

    latency_filename = (
        base_dir
        + f"{application}-{fault_config_file_name}-{phase}-{role}-Latency.csv"
    )

    with open(latency_filename, "w", encoding="UTF8", newline="") as csv_output:
        writer = csv.writer(csv_output)
        header = [
            "request_number", "replication_mode", "quorum_mode",
            "request_time_ms", "request_received_time_ms",
            "response_time_ms", "response_received_time_ms",
            "end_to_end_latency", "compute_time", "transmission_time",
            "timeout_threshold_ms", "detected_fault",
            "state_version", "applied",
        ]
        writer.writerow(header)

        for idx, result in enumerate(results, start=1):
            grpc_res = result["grpc_res"]
            end_to_end_latency = grpc_res.response_received_time_ms - grpc_res.request_time_ms
            compute_time = grpc_res.response_time_ms - grpc_res.request_received_time_ms
            transmission_time = end_to_end_latency - compute_time
            row = [
                idx,
                configs.REPLICATION_MODE,
                getattr(configs, "QUORUM_MODE", "NA"),
                grpc_res.request_time_ms,
                grpc_res.request_received_time_ms,
                grpc_res.response_time_ms,
                getattr(grpc_res, "response_received_time_ms", -1),
                end_to_end_latency,
                compute_time,
                transmission_time,
                result.get("timeout_threshold_ms", -1),
                result.get("detected_fault", False),
                getattr(grpc_res, "state_version", -1),
                getattr(grpc_res, "applied", True),
            ]
            writer.writerow(row)




def save_experiment_results(device_num, application, fault_config_file_name, results, res_utilizations):
    base_dir = (
        configs.PROJECT_PATH
        + "EDB/results/"
        + f"{configs.DEVICE_TYPE}_"
        + configs.EDGE_DEVICE_NAME.value
        + str(device_num)
        + f"/{configs.REPLICATION_MODE}_{configs.QUORUM_MODE}/"   # ✅ add replication mode + quorum
    )
    os.makedirs(base_dir, exist_ok=True)

    latency_filename = base_dir + application + "-" + fault_config_file_name + ".csv"
    print("********[x]***** Saving results for filename:{}".format(latency_filename))
    print("********[x]***** Size of results: " + str(len(results)))
    with open(latency_filename, "w", encoding="UTF8", newline="") as csv_output:
        writer = csv.writer(csv_output)

        header = [
            "experiment_id",
            "replication_mode",
            "quorum_mode",
            "request_time_ms",
            "request_received_time_ms",
            "response_time_ms",
            "response_received_time_ms",
            "end_to_end_latency",
            "compute_time",
            "transmission_time",
            "avg_cpu",
            "avg_memory",
            "avg_disk",
            "avg_network_received_sp",
            "avg_network_transmitted_sp",
            "avg_temperature",
            "state_version",
            "applied",
        ]
        writer.writerow(header)

        for idx, (grpc_res, res_utilization) in enumerate(zip(results, res_utilizations), start=1):
            end_to_end_latency = grpc_res.response_received_time_ms - grpc_res.request_time_ms
            compute_time = grpc_res.response_time_ms - grpc_res.request_received_time_ms
            transmission_time = end_to_end_latency - compute_time

            row = [
                idx,
                configs.REPLICATION_MODE,
                getattr(configs, "QUORUM_MODE", "NA"),
                grpc_res.request_time_ms,
                grpc_res.request_received_time_ms,
                grpc_res.response_time_ms,
                getattr(grpc_res, "response_received_time_ms", -1),
                end_to_end_latency,
                compute_time,
                transmission_time,
                res_utilization.average_cpu_utilization,
                res_utilization.average_memory_utilization,
                res_utilization.average_disk_utilization,
                res_utilization.average_network_received_speed,
                res_utilization.average_network_transmitted_speed,
                res_utilization.average_power_consumption,
                getattr(grpc_res, "state_version", -1),
                getattr(grpc_res, "applied", True),
            ]
            writer.writerow(row)


def save_resource_logs(device_num, application, fault_config_file_name, resource_logs, role="primary"):
    base_path = (
        configs.PROJECT_PATH
        + "EDB/results_over_time/"
        + f"{configs.DEVICE_TYPE}_"
        + configs.EDGE_DEVICE_NAME.value
        + str(device_num)
        + f"/{configs.REPLICATION_MODE}_{configs.QUORUM_MODE}/"
    )
    os.makedirs(base_path, exist_ok=True)

    cpu_filename = base_path + application + "-" + fault_config_file_name + f"-{role}" + "-CPU.txt"
    mem_filename = base_path + application + "-" + fault_config_file_name + f"-{role}" + "-MEM.txt"
    net_filename = base_path + application + "-" + fault_config_file_name + f"-{role}" + "-NET.txt"
    io_filename  = base_path + application + "-" + fault_config_file_name + f"-{role}" + "-IO.txt"
    cpu_temps_filename = base_path + application + "-" + fault_config_file_name + f"-{role}" + "-TEMP.txt"
    fault_injection_filename = base_path + application + "-" + fault_config_file_name + f"-{role}" + "-FaultInjection.csv"

    with open(cpu_filename, "wb") as cpu_f:
        cpu_f.write(resource_logs.cpu_log.data)

    with open(mem_filename, "wb") as mem_f:
        mem_f.write(resource_logs.memory_log.data)

    with open(net_filename, "wb") as net_f:
        net_f.write(resource_logs.network_log.data)

    with open(io_filename, "wb") as io_f:
        io_f.write(resource_logs.io_log.data)

    with open(cpu_temps_filename, 'w', encoding='UTF8', newline='') as temp_csv_output:
        # create the csv writer
        temp_writer = csv.writer(temp_csv_output)
        temp_writer.writerow(["Timestamp_ms", "CPU_Temp"])
        for t_time, cpu_t in zip(resource_logs.temperature_timestamps_ms, resource_logs.cpu_temperatures):
            temp_writer.writerow([t_time, cpu_t])

    with open(fault_injection_filename, 'w', encoding='UTF8', newline='') as fi_csv_output:
        fi_writer = csv.writer(fi_csv_output)

        fi_writer.writerow(["fault_injection_start_time", "fault_injection_stop_time"])
        for f_start in resource_logs.fault_injection_start_times_ms:
            fi_writer.writerow([f_start])

####################################################################################
####################################################################################
####################   TIMEOUT HANDLING    #########################################
####################################################################################
def detect_fault_on_timeout(result, timeout_threshold_ms):
    latency = result.response_received_time_ms - result.request_time_ms
    if latency > timeout_threshold_ms:
        return True  # Detected a timeout (fault)
    return False

####################################################################################
####################################################################################
####################################################################################
####################################################################################
def run_single_experiment(client, application, fault, fault_config, experiment_id):
    fault_config_file_name = 'no-fault'
    if fault is not None:
        fault_config_file_name = '{0}-{1}'.format(fault.abbreviation, fault_config)
    print("***************************************************************************************")
    print("****************** Starting experiment: device:{0} - ip:{1} - app:{2} - {3} - exp id: {4}".format(
        configs.EDGE_DEVICE_NAME.value, client.host, application, fault_config_file_name, experiment_id))
    print("***************************************************************************************")

    fault_injection_status = client.call_server_to_get_fault_injection_status()
    while not fault_injection_status.is_finished:
        print("[x] Previous experiment still in progress, we need to wait!! - (fault injection in progress)")
        time.sleep(3)
        fault_injection_status = client.call_server_to_get_fault_injection_status()
    print("********[x]***** Ready to start the experiment.")
    client.call_edge_to_start_resource_tracing()
    if fault is not None and fault.abbreviation != 'TCP' and fault.abbreviation != 'PING':
        client.call_server_to_inject_fault(fault.fault_command, fault_config)
    time.sleep(configs.TIME_BOUND_FOR_FAULT_INJECTION)  ### Sleep a bit until stressors are ready to go

    # **** Starting the experiment ****************** #
    grpc_result = get_application_result(application)
    resource_utilization_response = client.get_resource_utilization()
    return grpc_result, resource_utilization_response


####################################################################################
####################################################################################
def get_application_result(application_to_test):
    if application_to_test == 'MM':
        grpc_result = replicate_request(rm, "MM", lambda c: c.call_matrix_multiplication())
    elif application_to_test == 'FFT':
        grpc_result = replicate_request(rm, "FFT", lambda c: c.call_fast_fourier_transform())
    elif application_to_test == 'FPO-SIN':
        grpc_result = replicate_request(rm, "FPO-SIN", lambda c: c.call_floating_point_sine())
    elif application_to_test == 'FPO-SQRT':
        grpc_result = replicate_request(rm, "FPO-SQRT", lambda c: c.call_floating_point_sqrt())
    elif application_to_test == 'SORT':
        grpc_result = replicate_request(rm, "SORT", lambda c: c.call_sort_file())
    elif application_to_test == 'DD':
        grpc_result = replicate_request(rm, "DD", lambda c: c.call_dd_cmd())
    elif application_to_test == 'IPERF':
        grpc_result = replicate_request(rm, "IPERF", lambda c: c.call_iperf())
    elif application_to_test == 'IP':
        grpc_result = replicate_request(rm, "IP", lambda c: c.call_image_processing())
    elif application_to_test == 'SA':
        grpc_result = replicate_request(rm, "SA", lambda c: c.call_sentiment_analysis())
    elif application_to_test == 'ST':
        grpc_result = replicate_request(rm, "ST", lambda c: c.call_speech_to_text())
    elif application_to_test == 'IC-A-CPU':
        grpc_result = replicate_request(rm, "IC-A-CPU", lambda c: c.call_image_classification_alexnet_cpu())
    elif application_to_test == 'IC-A-GPU':
        grpc_result = replicate_request(rm, "IC-A-GPU", lambda c: c.call_image_classification_alexnet_gpu())
    elif application_to_test == 'IC-S-CPU':
        grpc_result = replicate_request(rm, "IC-S-CPU", lambda c: c.call_image_classification_squeezenet_cpu())
    elif application_to_test == 'IC-S-GPU':
        grpc_result = replicate_request(rm, "IC-S-GPU", lambda c: c.call_image_classification_squeezenet_gpu())
    elif application_to_test == 'OD-CPU':
        grpc_result = replicate_request(rm, "OD-CPU", lambda c: c.call_object_detection_darknet())
    elif application_to_test == 'OD-GPU':
        grpc_result = replicate_request(rm, "OD-GPU", lambda c: c.call_object_detection_darknet_gpu())
    elif application_to_test == 'PS':
        grpc_result = replicate_request(rm, "PS", lambda c: c.call_pocket_sphinx())
    elif application_to_test == 'AE':
        grpc_result = replicate_request(rm, "AE", lambda c: c.call_aeneas())
    elif application_to_test == 'OT-CPU':
        grpc_result = replicate_request(rm, "OT-CPU", lambda c: c.call_object_tracking())
    elif application_to_test == 'SA-AGG':
        req_id = str(uuid.uuid4())
        input_text = utils.random_sentence()
        upd = pb2.SentimentAggregationRequest(
            key="topicA",
            input_text=input_text,
            req_id=req_id,
            request_time_ms= utils.current_milli_time()
        )
        grpc_result = replicate_request(
            rm, "SA-AGG",
            lambda c: c.call_sentiment_aggregation(upd.key, upd.input_text, upd.req_id, upd.request_time_ms)
        )
    else:
        grpc_result = None

    if grpc_result:
        response_received_time_ms = utils.current_milli_time()
        # not all protos have this field; guard with hasattr
        if hasattr(grpc_result, "response_received_time_ms"):
            grpc_result.response_received_time_ms = response_received_time_ms
        logging.info(f"[x] Received Result of Application Call {application_to_test}")
    return grpc_result



####################################################################################
####################################################################################
def run_application_over_time_fault_free(edge_server, application_to_test, timeout_threshold_ms):
    print("***************************************************************************************")
    print("****************** Starting experiment: device:{0} - ip:{1} - app:{2} - Fault - Free - timeout:{3}".format(
        configs.EDGE_DEVICE_NAME.value, edge_server.host, application_to_test, timeout_threshold_ms))
    print("***************************************************************************************")
    # resource_tracing_status = edge_server.call_server_to_get_resource_tracking_status()
    # while not resource_tracing_status.is_finished:
    #     print("[x] Previous experiment still in progress, we need to wait!! - (resource monitoring in progress)")
    #     time.sleep(3)
    #     resource_tracing_status = edge_server.call_server_to_get_resource_tracking_status()
    edge_server.call_edge_to_start_resource_tracing_with_saving()
    # # wait for resource tracing to start
    # time.sleep(2)

    exp_results = []

    print("[x]**** Start Fault Free Operations")
    configs.EXPERIMENT_DURATION = configs.NUMBER_OF_FAULT_FREE_ROUNDS * configs.FAULT_FREE_DURATIONS
    start_time = time.time()
    while time.time() < start_time + configs.EXPERIMENT_DURATION:
        grpc_result = get_application_result(application_to_test)
        if grpc_result:
            # Now, use the timeout for detection
            detected_fault = detect_fault_on_timeout(grpc_result, timeout_threshold_ms)
            result_row = {
                'grpc_res': grpc_result,
                'timeout_threshold_ms': timeout_threshold_ms,
                'detected_fault': detected_fault,
            }
            exp_results.append(result_row)
    print("[x]**** End Fault Free Operations")
    return exp_results

def run_application_over_time(edge_server, application_to_test, fault_to_inject, fconfig):
    print("***************************************************************************************")
    print("****************** Starting experiment: device:{0} - ip:{1} - app:{2} - fault:{3} - config:{4}".format(
        configs.EDGE_DEVICE_NAME.value, edge_server.host, application_to_test, fault_to_inject.abbreviation, fconfig))
    print("***************************************************************************************")
    # resource_tracing_status = client.call_server_to_get_resource_tracking_status()
    # while not resource_tracing_status.is_finished:
    #     print("[x] Previous experiment still in progress, we need to wait!! - (fault injection in progress)")
    #     time.sleep(3)
    #     resource_tracing_status = client.call_server_to_get_resource_tracking_status()

    edge_server.call_edge_to_start_resource_tracing_with_saving()

    exp_results = []

    injected_faults_count = 0
    while injected_faults_count < configs.NUMBER_OF_FAULT_INJECTIONS:
        print("[x]**** Start Fault Free Operations")
        start_time = time.time()
        while time.time() < start_time + configs.FAULT_FREE_DURATIONS:
            grpc_result = get_application_result(application_to_test)
            if grpc_result:
                exp_results.append(grpc_result)
        print("[x]**** End Fault Free Operations")
        edge_server.call_server_to_inject_fault(fault_to_inject.fault_command, fconfig)
        print("[x]**** Start Faultyyyyy Operations")
        start_time = time.time()
        while time.time() < start_time + configs.FAULT_INJECTION_DURATION:
            grpc_result = get_application_result(application_to_test)
            if grpc_result:
                exp_results.append(grpc_result)
        print("[x]**** End Faultyyyyy Operations")
        # edge_server.call_server_to_stop_fault_injection()
        fault_injection_status = edge_server.call_server_to_get_fault_injection_status()
        while not fault_injection_status.is_finished:
            print("[x] Previous experiment still in progress, we need to wait!! - (fault injection in progress)")
            time.sleep(3)
            fault_injection_status = edge_server.call_server_to_get_fault_injection_status()
        injected_faults_count += 1

    print("[x]**** Start Fault Free Operations")
    start_time = time.time()
    while time.time() < start_time + configs.FAULT_FREE_DURATIONS:
        grpc_result = get_application_result(application_to_test)
        if grpc_result:
            exp_results.append(grpc_result)
    print("[x]**** End Fault Free Operations")
    return exp_results

    # sent_fault_request = False
    # while time.time() < start_time + configs.MAX_EXPERIMENT_TIME:
    #     if not sent_fault_request and time.time() > start_time + configs.FAULT_FREE_DURATIONS:
    #         edge_server.call_server_to_inject_fault(fault_to_inject.fault_command, fconfig)
    #         sent_fault_request = True
    #     grpc_result = get_application_result(application_to_test)
    #     if grpc_result:
    #         exp_results.append(grpc_result)


def run_application_over_time_with_timeout(edge_server, application_to_test, fault_to_inject, fconfig, timeout_threshold_ms):
    print("***************************************************************************************")
    print("****************** Starting experiment: device:{0} - ip:{1} - app:{2} - fault:{3} - config:{4} - timeoutThres:{5}".format(
        configs.EDGE_DEVICE_NAME.value, edge_server.host, application_to_test, fault_to_inject.abbreviation, fconfig, timeout_threshold_ms))
    print("***************************************************************************************")
    # resource_tracing_status = client.call_server_to_get_resource_tracking_status()
    # while not resource_tracing_status.is_finished:
    #     print("[x] Previous experiment still in progress, we need to wait!! - (fault injection in progress)")
    #     time.sleep(3)
    #     resource_tracing_status = client.call_server_to_get_resource_tracking_status()

    edge_server.call_edge_to_start_resource_tracing_with_saving()

    exp_results = []

    injected_faults_count = 0
    while injected_faults_count < configs.NUMBER_OF_FAULT_INJECTIONS:
        # print("[x]**** Start Fault Free Operations")
        # start_time = time.time()
        # while time.time() < start_time + configs.FAULT_FREE_DURATIONS:
        #     grpc_result = get_application_result(application_to_test)
        #     if grpc_result:
        #         # Now, use the timeout for detection
        #         detected_fault = detect_fault_on_timeout(grpc_result, timeout_threshold_ms)
        #         result_row = {
        #             'grpc_res': grpc_result,
        #             'timeout_threshold_ms': timeout_threshold_ms,
        #             'detected_fault': detected_fault,
        #         }
        #         exp_results.append(result_row)
        # print("[x]**** End Fault Free Operations")
        edge_server.call_server_to_inject_fault(fault_to_inject.fault_command, fconfig)
        print("[x]**** Start Faultyyyyy Operations")
        start_time = time.time()
        while time.time() < start_time + configs.FAULT_INJECTION_DURATION:
            grpc_result = get_application_result(application_to_test)
            if grpc_result:
                # Now, use the timeout for detection
                detected_fault = detect_fault_on_timeout(grpc_result, timeout_threshold_ms)
                result_row = {
                    'grpc_res': grpc_result,
                    'timeout_threshold_ms': timeout_threshold_ms,
                    'detected_fault': detected_fault,
                }
                exp_results.append(result_row)
        print("[x]**** End Faultyyyyy Operations")
        # edge_server.call_server_to_stop_fault_injection()
        # fault_injection_status = client.call_server_to_get_fault_injection_status()
        # while not fault_injection_status.is_finished:
        #     print("[x] Previous experiment still in progress, we need to wait!! - (fault injection in progress)")
        #     time.sleep(3)
        #     fault_injection_status = client.call_server_to_get_fault_injection_status()
        injected_faults_count += 1

    # print("[x]**** Start Fault Free Operations")
    # start_time = time.time()
    # while time.time() < start_time + configs.FAULT_FREE_DURATIONS:
    #     grpc_result = get_application_result(application_to_test)
    #     if grpc_result:
    #         # Now, use the timeout for detection
    #         detected_fault = detect_fault_on_timeout(grpc_result, timeout_threshold_ms)
    #         result_row = {
    #             'grpc_res': grpc_result,
    #             'timeout_threshold_ms': timeout_threshold_ms,
    #             'detected_fault': detected_fault,
    #         }
    #         exp_results.append(result_row)
    # print("[x]**** End Fault Free Operations")
    return exp_results


def start_tracing_all():
    """Start resource tracing on all replicas in the group."""
    for replica in rm.replicas:
        try:
            replica.call_edge_to_start_resource_tracing_with_saving()
        except Exception as e:
            logging.warning(f"[Tracing] Failed to start on {replica.host}: {e}")

def fetch_logs_all(app, suffix, phase="normal", results=None):
    """
    Fetch logs from all replicas, tagging them with phase (precrash/postcrash/normal).
    Save latency results only for the primary.
    """
    for ridx, replica in enumerate(rm.replicas, start=1):
        role = "primary" if ridx - 1 == rm.primary_idx else "backup"
        try:
            if role == "primary" and results is not None:
                save_experiment_results_over_time(
                    ridx, app, suffix, results,
                    role=role, phase=phase
                )
            resource_logs = replica.get_resource_logs()
            save_resource_logs(ridx, app, f"{suffix}-{phase}-{role}", resource_logs, role=role)
        except Exception as e:
            logging.warning(f"[fetch_logs_all] Failed to fetch logs from {replica.host} ({role}): {e}")


def run_fault_free(app, round_id, ridx):
    primary_idx = rm.primary_idx
    primary_host = rm.replicas[primary_idx].host
    configs.FAIL_DETECT_TIMEOUT_MS = 10**9   # ✅ effectively disable failover

    if already_done(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                    None, None, round_id, replica_id=ridx + 1):
        logging.info(
            f"Skipping DONE: {configs.DEVICE_TYPE}-{app}-{configs.REPLICATION_MODE}-NoFault-R{round_id}-replica{ridx+1}"
        )
        return

    log_progress(
        configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
        None, None, round_id, "STARTED",
        replica_id=primary_idx + 1, replica_host=primary_host,
        phase="fault-free"
    )

    try:
        start_tracing_all()
        results = run_application_over_time_fault_free(rm.replicas[primary_idx], app, timeout_threshold_ms=configs.FAIL_DETECT_TIMEOUT_MS)

        suffix = f"NoFault-Round:{round_id}"
        save_experiment_results_over_time(primary_idx + 1, app, suffix, results)
        fetch_logs_all(app, suffix)

        log_progress(
            configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
            None, None, round_id, "DONE",
            replica_id=primary_idx + 1, replica_host=primary_host,
            phase="fault-free"
        )
    except Exception as e:
        log_progress(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                     None, None, round_id, f"FAILED:{e}",
                     replica_id=primary_idx + 1, replica_host=primary_host, phase="fault-free")
        raise



def run_with_fault(app, fault, fault_config, round_id, ridx):
    """
    Stress-only run (no crash):
      1) Start tracing on all replicas
      2) Inject stress (CPU/MEM/IO/...)
      3) Send requests for the entire fault window
      4) Collect latency/timeout/freshness data
      5) Save per-replica logs and the primary's latency dataset

    Suffix example:
      Stress-CPU-90-Timeout:XXX-Round:Y
    """
    primary_idx = rm.primary_idx
    primary_host = rm.replicas[primary_idx].host

    if already_done(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                    fault, fault_config, round_id, replica_id=ridx + 1):
        logging.info(
            f"Skipping DONE: {configs.DEVICE_TYPE}-{app}-{configs.REPLICATION_MODE}-{fault.abbreviation}-{fault_config}-R{round_id}-replica{ridx+1}"
        )
        return

    for timeout in configs.TIMEOUT_THRESHOLDS[app]:
        configs.FAIL_DETECT_TIMEOUT_MS = timeout
        try:
            log_progress(
                configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                fault, fault_config, round_id, "STARTED",
                replica_id=primary_idx + 1, replica_host=primary_host,
                phase="stress", timeout_threshold_ms=timeout
            )
            start_tracing_all()

            results = []
            edge_server = rm.replicas[rm.primary_idx]
            edge_server.call_server_to_inject_fault(fault.fault_command, fault_config)
            log_replication_event(
                "FAULT_INJECT_START",
                f"fault={fault.abbreviation},cfg={fault_config},on_host={edge_server.host}"
            )

            end_at = time.time() + configs.FAULT_INJECTION_DURATION
            while time.time() < end_at:
                grpc_result = get_application_result(app)
                if grpc_result:
                    detected_fault = detect_fault_on_timeout(grpc_result, timeout)
                    results.append({
                        "grpc_res": grpc_result,
                        "timeout_threshold_ms": timeout,
                        "detected_fault": detected_fault,
                    })
                time.sleep(getattr(configs, "REQ_PACE_MS", 50) / 1000.0)

            # Let the stressor finish gracefully
            try:
                status = edge_server.call_server_to_get_fault_injection_status()
                while not status.is_finished:
                    time.sleep(0.5)
                    status = edge_server.call_server_to_get_fault_injection_status()
            except Exception:
                pass
            log_replication_event(
                "FAULT_INJECT_END",
                f"fault={fault.abbreviation},cfg={fault_config}"
            )

            # --- Persist outputs
            suffix = f"Stress-{fault.abbreviation}-{fault_config}-Timeout:{timeout}-Round:{round_id}"
            save_experiment_results_over_time(primary_idx + 1, app, suffix, results)
            fetch_logs_all(app, suffix)

            log_progress(
                configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                fault, fault_config, round_id, "DONE",
                replica_id=primary_idx + 1, replica_host=primary_host,
                phase="stress", timeout_threshold_ms=timeout
            )
        except Exception as e:
            log_progress(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                         fault, fault_config, round_id, f"FAILED:{e}",
                         replica_id=primary_idx + 1, replica_host=primary_host, phase="stress", timeout_threshold_ms=timeout)
            raise


def run_checkpoint_sweep(app, round_id, ridx):
    primary_idx = rm.primary_idx
    primary_host = rm.replicas[primary_idx].host

    if already_done(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                    None, None, round_id, replica_id=ridx + 1):
        logging.info(
            f"Skipping DONE checkpoint sweep: {configs.DEVICE_TYPE}-{app}-{configs.REPLICATION_MODE}-R{round_id}-replica{ridx+1}"
        )
        return

    log_progress(
        configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
        None, None, round_id, "STARTED",
        replica_id=primary_idx + 1, replica_host=primary_host,
        phase="checkpoint"
    )

    try:
        # pick one calibrated timeout threshold for this app
        timeout = configs.TIMEOUT_THRESHOLDS[app][0]
        configs.FAIL_DETECT_TIMEOUT_MS = timeout

        for cp_period in configs.CHECKPOINT_PERIODS_SA:
            log_replication_event("SET_CHECKPOINT_PERIOD", f"seconds={cp_period}")
            rm.replicas[primary_idx].call_set_checkpoint_period(cp_period)

            start_tracing_all()
            results = run_application_over_time_fault_free(
                rm.replicas[primary_idx], app, timeout
            )

            suffix = f"Checkpoint{cp_period}s-NoFault-Timeout:{timeout}-Round:{round_id}"
            save_experiment_results_over_time(primary_idx + 1, app, suffix, results)
            fetch_logs_all(app, suffix)

            # Freshness check
            if results:
                leader_version = results[-1]["grpc_res"].state_version
                replica_versions = []
                for replica in rm.replicas:
                    try:
                        vresp = replica.call_get_current_version("topicA")
                        replica_versions.append(vresp.state_version)
                    except Exception:
                        continue
                if replica_versions:
                    max_lag = leader_version - min(replica_versions)
                    log_replication_event(
                        "FRESHNESS_LAG",
                        f"leader={leader_version},replicas={replica_versions},lag={max_lag}"
                    )

        log_progress(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                     None, None, round_id, "DONE",
                     replica_id=primary_idx + 1, replica_host=primary_host,
                     phase="checkpoint")
    except Exception as e:
        log_progress(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                     None, None, round_id, f"FAILED:{e}",
                     replica_id=primary_idx + 1, replica_host=primary_host, phase="checkpoint")
        raise



# --- Crash/Recovery Helpers -----------------------------------------------

import subprocess
import time
import logging

def kill_edge_manager(ip, user=None, ssh_key=None):
    user = user or getattr(configs, "SSH_USER", "pi")

    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{ip}",
        "pkill -9 -f edge_device_manager.py || true"
    ]
    if ssh_key:
        ssh_cmd[1:1] = ["-i", ssh_key]

    res = subprocess.run(
        ssh_cmd,
        check=False,  # don't raise
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if res.returncode in (0, 1, 255):  # 255 = remote connection dropped (expected)
        logging.warning(f"[CRASH-INJECTOR] Killed edge_device_manager on {ip} (exit={res.returncode})")
        log_replication_event("INJECT_CRASH", f"replica_ip={ip}, exit={res.returncode}")
        return True
    else:
        logging.error(f"[CRASH-INJECTOR] SSH truly failed for {ip}: {res.stderr.strip()}")
        return False

def restart_edge_manager(ip, user=None, ssh_key=None):
    """
    Restart the edge_device_manager.py process on a replica via SSH.
    Uses conda env's python and detaches cleanly.
    """
    user = user or getattr(configs, "SSH_USER", "pi")

    project_dir = "/home/pi/Projects/EDB/edge_manager"
    python_bin = "/home/pi/miniforge3/envs/benchmark/bin/python"
    log_file = "/home/pi/edge_manager.log"

    remote_cmd = (
        f"cd {project_dir} && "
        f"export PATH=/home/{device_type}/Projects/darknet:$PATH && "
        f"nohup {python_bin} edge_device_manager.py "
        f"> {log_file} 2>&1 < /dev/null &"
    )

    ssh_cmd = [
        "ssh",
        "-f",  # fork after authentication
        "-n",  # prevent ssh from reading stdin
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if ssh_key:
        ssh_cmd.extend(["-i", ssh_key])
    ssh_cmd.append(f"{user}@{ip}")
    ssh_cmd.append(remote_cmd)

    try:
        subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.info(f"[RECOVERY] Restarted edge_manager on {ip}")
        log_replication_event("RESTART_REPLICA", f"replica_ip={ip}")
        return True
    except Exception as e:
        logging.error(f"[RECOVERY] Failed to restart edge_manager on {ip}: {e}")
        return False


def pick_crash_target_index():
    """
    Which replica should we crash?
    - PASSIVE: crash the current primary to test promotion.
    - ACTIVE: crash a non-primary replica to test quorum robustness.
    """
    if configs.REPLICATION_MODE == "PASSIVE":
        return rm.primary_idx
    # ACTIVE: pick a different replica than the current primary
    for i in range(len(rm.replicas)):
        if i != rm.primary_idx:
            return i
    # Fallback (degenerate 1-node case)
    return rm.primary_idx


def wait_for_new_leader(timeout_s=15, poll_ms=300):
    """
    After a crash, find a responsive replica by calling get_current_version (SA-AGG).
    If the app is not SA-AGG you can still use ping to detect liveness—here we try version first,
    then fall back to ping.

    Returns (new_primary_idx or None)
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # Try all replicas; the first one that responds wins
        for i, replica in enumerate(rm.replicas):
            try:
                # # Prefer SA-AGG version RPC if available; fall back to ping otherwise
                # if "SA-AGG" in configs.APPLICATIONS:
                #     # We send a dummy key; service should ignore or handle it
                #     v = replica.call_get_current_version("topicA")
                #     if getattr(v, "state_version", None) is not None:
                #         rm.primary_idx = i  # adopt this as the new leader
                #         log_replication_event("NEW_LEADER_DETECTED", f"replica_idx={i},host={replica.host}")
                #         return i
                # Fallback ping
                replica.call_ping()
                rm.primary_idx = i
                log_replication_event("NEW_LEADER_DETECTED", f"replica_idx={i},host={replica.host},via=ping")
                return i
            except Exception:
                continue
        time.sleep(poll_ms / 1000.0)
    logging.error("[Recovery] No responsive replica detected within timeout")
    return None


def get_latest_state_version_safely():
    """
    Ask the *current* responsive leader for state_version (for SA-AGG).
    Returns integer version or None if not available.
    """
    try:
        v = rm.replicas[rm.primary_idx].call_get_current_version("topicA")
        return getattr(v, "state_version", None)
    except Exception:
        return None


def run_failure_recovery(app, fault, fault_config, round_id, ridx):
    """
    Controlled crash under stress:
      1) Start tracing on all replicas
      2) Inject stress (CPU/MEM/IO/...)
      3) Midway through the window, crash the target replica (depends on replication mode)
      4) Keep issuing requests to observe failover
      5) Detect new leader and compute lost updates (SA-AGG)
      6) Save per-replica logs and the primary's latency dataset
    """
    primary_idx = rm.primary_idx
    primary_host = rm.replicas[primary_idx].host

    if already_done(configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                    fault, fault_config, round_id, replica_id=ridx + 1):
        logging.info(
            f"Skipping DONE failure recovery: {configs.DEVICE_TYPE}-{app}-{configs.REPLICATION_MODE}-{fault.abbreviation}-{fault_config}-R{round_id}-replica{ridx+1}"
        )
        return

    for timeout in configs.TIMEOUT_THRESHOLDS[app]:
        configs.FAIL_DETECT_TIMEOUT_MS = timeout
        try:
            log_progress(
                configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                fault, fault_config, round_id, "STARTED",
                replica_id=primary_idx + 1, replica_host=primary_host, phase="failure_recovery",
                timeout_threshold_ms=timeout
            )
            suffix = f"FailureRecovery-{fault.abbreviation}-{fault_config}-Timeout:{timeout}-Round:{round_id}"

            start_tracing_all()
            results = []

            edge_server = rm.replicas[rm.primary_idx]
            edge_server.call_server_to_inject_fault(fault.fault_command, fault_config)
            log_replication_event(
                "FAULT_INJECT_START",
                f"fault={fault.abbreviation},cfg={fault_config},on_host={edge_server.host}"
            )

            total_dur = configs.FAULT_INJECTION_DURATION
            crash_at = time.time() + (total_dur / 2.0)
            end_at = time.time() + total_dur
            crashed = False
            pre_crash_version = None
            target_ip = None

            while time.time() < end_at:
                try:
                    grpc_result = get_application_result(app)
                except Exception as e:
                    if "UNAVAILABLE" in str(e):
                        logging.warning(f"[FAILOVER] Primary unavailable during crash injection: {e}")
                        rm.detect_failures()  # let heartbeat mark dead replica
                        time.sleep(0.5)
                        continue
                    else:
                        raise

                if grpc_result:
                    detected_fault = detect_fault_on_timeout(grpc_result, timeout)
                    results.append({
                        "grpc_res": grpc_result,
                        "timeout_threshold_ms": timeout,
                        "detected_fault": detected_fault,
                    })
                    if app == "SA-AGG":
                        pre_crash_version = grpc_result.state_version

                if (not crashed) and (time.time() >= crash_at):
                    target_idx = pick_crash_target_index()
                    target_ip = rm.replicas[target_idx].host
                    # Save pre-crash
                    save_experiment_results_over_time(primary_idx + 1, app, suffix, results, role="primary",
                                                      phase="precrash")
                    fetch_logs_all(app, suffix, phase="precrash", results=results)
                    ok = kill_edge_manager(target_ip, user=getattr(configs, "SSH_USER", "pi"))
                    crashed = True
                    log_replication_event(
                        "CRASH_TRIGGERED",
                        f"target_idx={target_idx},host={target_ip},ok={ok},mode={configs.REPLICATION_MODE}"
                    )

                    # ✅ Wait for new leader after crash
                    new_leader_idx = wait_for_new_leader(timeout_s=15)
                    if new_leader_idx is None:
                        log_replication_event("RECOVERY_FAILED", "no_new_leader_detected_after_crash")
                        break

                    # ✅ Reconfigure roles for passive SA-AGG only
                    if configs.REPLICATION_MODE == "PASSIVE" and app == "SA-AGG":
                        try:
                            _reconfigure_roles_for_passive(rm)
                            logging.info(f"[role] Reconfigured roles after failover (new_primary={new_leader_idx})")
                        except Exception as e:
                            log_replication_event(
                                "NEW_PRIMARY_FAILED",
                                f"new_primary_idx={rm.primary_idx},host={rm.replicas[rm.primary_idx].host}"
                            )
                            logging.warning(f"[role] reconfigure after failover failed: {e}")
                    log_replication_event(
                        "NEW_PRIMARY_READY",
                        f"new_primary_idx={rm.primary_idx},host={rm.replicas[rm.primary_idx].host}"
                    )

                time.sleep(getattr(configs, "REQ_PACE_MS", 50) / 1000.0)

            # Let stressor finish
            try:
                status = edge_server.call_server_to_get_fault_injection_status()
                while not status.is_finished:
                    time.sleep(0.5)
                    status = edge_server.call_server_to_get_fault_injection_status()
            except Exception:
                pass
            log_replication_event(
                "FAULT_INJECT_END",
                f"fault={fault.abbreviation},cfg={fault_config}"
            )

            if app == "SA-AGG" and pre_crash_version is not None:
                new_leader_idx = wait_for_new_leader(timeout_s=15)
                if new_leader_idx is not None:
                    post_crash_version = get_latest_state_version_safely()
                    if post_crash_version is not None:
                        lost = max(0, pre_crash_version - post_crash_version)
                        log_replication_event(
                            "LOST_UPDATES",
                            f"lost={lost},pre={pre_crash_version},post={post_crash_version},"
                            f"new_leader_idx={new_leader_idx},host={rm.replicas[new_leader_idx].host}"
                        )
                else:
                    log_replication_event("RECOVERY_FAILED", "no_new_leader_detected")

            # Save post-crash
            save_experiment_results_over_time(rm.primary_idx + 1, app, suffix, results, role="primary",
                                              phase="postcrash")
            fetch_logs_all(app, suffix, phase="postcrash", results=results)

            # At the very end of run_failure_recovery
            if crashed and target_ip:
                logging.info(f"[RECOVERY] Restarting crashed replica at {target_ip}")
                restart_edge_manager(target_ip, user=getattr(configs, "SSH_USER", "pi"))
                # small sleep to let the replica rejoin
                time.sleep(50)

            log_progress(
                configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                fault, fault_config, round_id, "DONE",
                replica_id=primary_idx + 1, replica_host=primary_host, phase="failure_recovery", timeout_threshold_ms=timeout
            )

        except Exception as e:
            log_progress(
                configs.DEVICE_TYPE, app, configs.REPLICATION_MODE, configs.QUORUM_MODE,
                fault, fault_config, round_id, f"FAILED:{e}",
                replica_id=primary_idx + 1, replica_host=primary_host, phase="failure_recovery", timeout_threshold_ms=timeout
            )
            raise




####################################################################################
####################################################################################
if __name__ == '__main__':
    utils.initial_workload_setup()

    # for device_type, group in [("nano", configs.NANO_GROUP), ("pi", configs.PI_GROUP)]:
    for device_type, group in [("pi", configs.PI_GROUP)]:
        configs.DEVICE_TYPE = device_type
        if device_type == "pi":
            configs.EDGE_DEVICE_NAME = configs.EdgeDevice.RPI
        else:
            configs.EDGE_DEVICE_NAME = configs.EdgeDevice.NANO

        # for repl_mode in ["BASELINE", "PASSIVE", "ACTIVE"]:
        # for repl_mode in ["PASSIVE", "ACTIVE"]:
        for repl_mode in ["PASSIVE"]:
            configs.REPLICATION_MODE = repl_mode
            quorum_modes = ["MAJORITY"] if repl_mode == "ACTIVE" else ["NA"]

            global rm
            rm = ReplicaManager(group)
            if repl_mode != "BASELINE":
                threading.Thread(target=rm.heartbeat_loop, daemon=True).start()
                logging.info(f"ReplicaManager heartbeat started for {device_type}")
            else:
                logging.info(f"ReplicaManager created for {device_type} in BASELINE (no heartbeat loop)")

            for quorum in quorum_modes:
                configs.QUORUM_MODE = quorum

                for app in configs.APPLICATIONS:
                    configs.CURRENT_APP = app
                    if device_type == "pi" and app == "OD-GPU":
                        continue
                    logging.info(f"=== {device_type} app={app} mode={repl_mode} quorum={quorum} ===")

                    # # --- Step 1: Fault-free runs (ensure data exists for calibration)
                    # for round_id in range(configs.REPETITIONS):
                    #     configs.CURRENT_FAULT = "NA"
                    #     if repl_mode in ["BASELINE", "PASSIVE"]:
                    #         # rotate primaries
                    #         for ridx in range(len(rm.replicas)):
                    #             rm.primary_idx = ridx
                    #             if repl_mode == "PASSIVE" and app == "SA-AGG":
                    #                 _reconfigure_roles_for_passive(rm)
                    #             logging.info(
                    #                 f"[{device_type}] Fault-free {app} {repl_mode}, replica {ridx} ({rm.replicas[ridx].host}) as primary")
                    #             run_fault_free(app, round_id, ridx)
                    #     else:  # ACTIVE
                    #         run_fault_free(app, round_id, 0)

                    # --- Step 2: Calibrate timeouts from just-produced fault-free CSVs
                    calibrate_timeouts.calibrate_timeouts(app, repl_mode, quorum)

                    # # --- Step 3: Faulty (stress-only) runs
                    # for fault in configs.FAULTS:
                    #     configs.CURRENT_FAULT = fault.fault_name
                    #     for fault_config in fault.fault_config:
                    #         for round_id in range(configs.REPETITIONS):
                    #             if repl_mode in ["BASELINE", "PASSIVE"]:
                    #                 for ridx in range(len(rm.replicas)):
                    #                     rm.primary_idx = ridx
                    #                     if repl_mode == "PASSIVE" and app == "SA-AGG":
                    #                         _reconfigure_roles_for_passive(rm)
                    #                     logging.info(
                    #                         f"[{device_type}] Faulty {app} {repl_mode}, {fault.abbreviation}-{fault_config}, replica {ridx} as primary")
                    #                     run_with_fault(app, fault, fault_config, round_id, ridx)
                    #             else:  # ACTIVE
                    #                 run_with_fault(app, fault, fault_config, round_id, 0)

                    # # --- Step 4: Checkpoint sweeps (SA-AGG only)
                    # if app == "SA-AGG" and repl_mode in ["PASSIVE", "ACTIVE"]:
                    #     configs.CURRENT_FAULT = "NA"
                    #     for round_id in range(configs.REPETITIONS):
                    #         if repl_mode == "PASSIVE":
                    #             for ridx in range(len(rm.replicas)):
                    #                 rm.primary_idx = ridx
                    #                 _reconfigure_roles_for_passive(rm)
                    #                 logging.info(
                    #                     f"[{device_type}] SA-AGG checkpoint sweep replica {ridx} ({rm.replicas[ridx].host}) as primary")
                    #                 run_checkpoint_sweep(app, round_id, ridx)
                    #         elif repl_mode == "ACTIVE":
                    #             run_checkpoint_sweep(app, round_id, 0)

                    # --- Step 5: Failure recovery runs
                    if app in ["SA-AGG", "IP"]:
                        for fault in configs.FAULTS:
                            configs.CURRENT_FAULT = fault.fault_name
                            for fault_config in fault.fault_config:
                                for round_id in range(configs.REPETITIONS):
                                    if repl_mode in ["PASSIVE"]:
                                        for ridx in range(len(rm.replicas)):
                                            rm.primary_idx = ridx
                                            if repl_mode == "PASSIVE" and app == "SA-AGG":
                                                _reconfigure_roles_for_passive(rm)
                                            logging.info(
                                                f"[{device_type}] Failure recovery {app} {repl_mode}, {fault.abbreviation}-{fault_config}, replica {ridx} as primary")
                                            run_failure_recovery(app, fault, fault_config, round_id, ridx)
                                    elif repl_mode == "ACTIVE":  # ACTIVE
                                        run_failure_recovery(app, fault, fault_config, round_id, 0)

# if __name__ == '__main__':
#     utils.initial_workload_setup()
#
#     # -----------------------------------------------------
#     # Initialize Replica Manager for stateful SA-AGG service
#     # -----------------------------------------------------
#     # --------------------------------------
#     # Initialize ReplicaManager (all services)
#     # --------------------------------------
#     global rm
#     rm = ReplicaManager("SA-AGG")  # main service name
#     threading.Thread(target=rm.heartbeat_loop, daemon=True).start()
#     logging.info("ReplicaManager heartbeat started")
#
#     device_num = 2
#     ##############################################################
#     ########## Running applications with Timeout #################
#     ##############################################################
#     for edge_device_ip in configs.EDGE_DEVICES_IP:
#         client = grpc_client.Client(edge_device_ip)
#         for app in configs.APPLICATIONS:
#             for fault in configs.FAULTS:
#                 for fault_config in fault.fault_config:
#                     for timeout in configs.TIMEOUT_THRESHOLDS[app]:
#                         for experiment_round in range(configs.REPETITIONS):
#                             print(f"Running {app} with fault {fault.abbreviation}-{fault_config} and timeout {timeout}ms round:{experiment_round}")
#                             # Run the experiment (could be run_application_over_time or your custom function)
#                             experiment_results = run_application_over_time_with_timeout(client, app, fault, fault_config, timeout)
#                             # Save results, including timeout value in filename and data
#                             save_experiment_results_over_time(device_num, app,
#                                                                  f"{fault.abbreviation}-{fault_config}-Timeout:{timeout}-Round:{experiment_round}",
#                                                                  experiment_results)
#                             resource_logs = client.get_resource_logs()
#                             save_resource_logs(device_num, app, f"{fault.abbreviation}-{fault_config}-Timeout:{timeout}-Round:{experiment_round}",
#                                                resource_logs)
#         device_num += 1
#
#     device_num = 2
#     for edge_device_ip in configs.EDGE_DEVICES_IP:
#         client = grpc_client.Client(edge_device_ip)
#         for app in configs.APPLICATIONS:
#             for timeout in configs.TIMEOUT_THRESHOLDS[app]:
#                 for experiment_round in range(configs.REPETITIONS):
#                     ######################################################################
#                     ####### Fault Free Resource Evaluations ##############################
#                     experiment_results = run_application_over_time_fault_free(client, app, timeout)
#                     time.sleep(10)
#                     save_experiment_results_over_time(device_num, app, f"No-Fault-Timeout:{timeout}-Round:{experiment_round}", experiment_results)
#                     resource_logs = client.get_resource_logs()
#                     save_resource_logs(device_num, app, f"No-Fault-Timeout:{timeout}-Round:{experiment_round}", resource_logs)
#                     ######################################################################
#         device_num += 1

        # for app in configs.APPLICATIONS:
        #     for fault in configs.FAULTS:
        #         for fault_config in fault.fault_config:
        #             experiment_results = run_application_over_time(client, app, fault, fault_config)
        #             time.sleep(10)
        #             save_experiment_results_over_time(app, '{0}-{1}'.format(fault.abbreviation, fault_config),
        #                                               experiment_results)
        #             resource_logs = client.get_resource_logs()
        #             save_resource_logs(app, '{0}-{1}'.format(fault.abbreviation, fault_config), resource_logs)


            ################################################################################
            ################################################################################
            ################################################################################

            # experiment_results = []
            # resource_utilizations = []
            # # Fault Free Experiments
            # experiment_id = 1
            # while experiment_id <= configs.REPEAT_EXPERIMENTS:
            #     exp_result, resource_utilization = run_single_experiment(client, application, None, None,
            #                                                               experiment_id)
            #     experiment_results.append(exp_result)
            #     resource_utilizations.append(resource_utilization)
            #     experiment_id += 1
            #
            # print("********[x]***** Saving experiment results")
            # save_experiment_results(application, "no-fault", experiment_results, resource_utilizations)

            # Experiments with Fault Injection
            # for fault in configs.FAULTS:
            #     for fault_config in fault.fault_config:
            #         experiment_results = []
            #         resource_utilizations = []
            #         experiment_id = 1
            #         while experiment_id <= configs.REPEAT_EXPERIMENTS:
            #             exp_result, resource_utilization = run_single_experiment(client, application, fault,
            #                                                                      fault_config, experiment_id)
            #             experiment_results.append(exp_result)
            #             resource_utilizations.append(resource_utilization)
            #             experiment_id += 1
            #
            #         print("********[x]***** Saving experiment results - with fault injections")
            #         save_experiment_results(application, '{0}-{1}'.format(fault.abbreviation, fault_config),
            #                                 experiment_results, resource_utilizations)
            #
            # analyze_results.analyze_result_for_application(application, edge_device_ip)
