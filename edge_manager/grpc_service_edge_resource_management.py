import utils
from utils import *
from protos import benchmark_pb2_grpc as pb2_grpc
from protos import benchmark_pb2 as pb2

import psutil
import subprocess
import multiprocessing
import threading
import time
import os
import signal


class EdgeResourceManagementGRPCService(pb2_grpc.EdgeResourceManagementServicer):

    def ping(self, request, context):
        return pb2.EmptyProto()

    def __init__(self, *args, **kwargs):
        self.resource_thread = None
        self.power_thread = None
        self.fault_injection_process = None
        self.fault_injection_parent_process = None
        self.fault_injection_start_times_ms = []
        self.fault_injection_stop_times_ms = []
        self.utilization_output = None
        self.resource_tracing_process = None

    # ------------------------------
    # helpers to safely restart threads
    # ------------------------------
    def _stop_existing_threads(self):
        # Stop power thread if alive
        if self.power_thread:
            try:
                self.power_thread.stop()
                self.power_thread.join()
                print("[x] Old Power thread stopped.")
            except Exception as e:
                print(f"[!] Error stopping old power thread: {e}")
            self.power_thread = None

        # Stop resource thread if alive
        if self.resource_thread:
            try:
                self.resource_thread.stop()
                self.resource_thread.join()
                print("[x] Old Resource thread stopped.")
            except Exception as e:
                print(f"[!] Error stopping old resource thread: {e}")
            self.resource_thread = None

    # ------------------------------
    # gRPC methods
    # ------------------------------
    def start_resource_tracing(self, request, context):
        print("[x] Tracing resource utilization.")
        # clean up previous threads before starting new ones
        self._stop_existing_threads()

        # Start fresh threads
        self.resource_thread = ResourceUtilizationThread(interval=1)
        self.resource_thread.start()

        self.power_thread = PowerMeasurementThread(interval=1)
        self.power_thread.start()

        return pb2.EmptyProto()

    def start_resource_tracing_and_saving(self, request, context):
        print("[x] Tracing resource utilization with saving.")
        # clean up previous threads before starting new ones
        self._stop_existing_threads()

        # Start fresh threads
        self.resource_thread = ResourceUtilizationSavingThread(
            interval=1, timeout=request.timeout
        )
        self.resource_thread.start()

        self.power_thread = PowerMeasurementThread(interval=1)
        self.power_thread.start()

        return pb2.EmptyProto()

    def get_resource_utilization(self, request, context):
        # Stop the power measurement thread and get the average power consumption
        self.power_thread.stop()
        self.power_thread.join()
        # Stop the resource utilization thread and get the average values
        self.resource_thread.stop()
        self.resource_thread.join()
        # Send a signal to the stress process to terminate it
        if self.fault_injection_process:
            os.kill(self.fault_injection_process.pid, signal.SIGTERM)

        avg_power = self.power_thread.get_average_power()
        avg_cpu_utilization = self.resource_thread.get_average_cpu_utilization()
        avg_memory_utilization = self.resource_thread.get_average_memory_utilization()
        avg_disk_utilization = self.resource_thread.get_average_disk_utilization()
        avg_network_received_speed, avg_network_transmitted_speed = self.resource_thread.get_average_network_utilization()
        resource_utilization_response = pb2.ResourceUtilizationResponse()
        resource_utilization_response.average_cpu_utilization = avg_cpu_utilization
        resource_utilization_response.average_memory_utilization = avg_memory_utilization
        resource_utilization_response.average_disk_utilization = avg_disk_utilization
        resource_utilization_response.average_network_received_speed = avg_network_received_speed
        resource_utilization_response.average_network_transmitted_speed = avg_network_transmitted_speed
        resource_utilization_response.average_power_consumption = avg_power
        return resource_utilization_response

    def get_fault_injection_status(self, request, context):
        if self.fault_injection_process is None:
            # No faults has been injected yet
            return pb2.ProcessStatus(is_finished=True)
        poll = self.fault_injection_process.poll()
        if poll is None:
            # A None value indicates that the process hasn't terminated yet
            print("[xxxx] Fault Injection Still in Process")
            return pb2.ProcessStatus(is_finished=False)
        else:
            return pb2.ProcessStatus(is_finished=True)

    def get_resource_tracing_status(self, request, context):
        if self.resource_thread is None:
            # No resource tracing has been done yet
            return pb2.ProcessStatus(is_finished=True)
        self.resource_thread.stop()
        poll_cpu = self.resource_thread.get_cpu_process().poll()
        poll_memory = self.resource_thread.get_memory_process().poll()
        poll_network = self.resource_thread.get_network_process().poll()
        poll_io = self.resource_thread.get_io_process().poll()

        if poll_cpu is None or poll_memory is None or poll_network is None or poll_io is None:
            # A None value indicates that the process hasn't terminated yet
            print("[xxxx] Resource Tracing is in Process")
            return pb2.ProcessStatus(is_finished=False)
        else:
            return pb2.ProcessStatus(is_finished=True)

    def inject_fault(self, request, context):
        fault_command = request.fault_command
        fault_config = request.fault_config
        stress_string = 'stress-ng {0} {1} --timeout 60'
        shell_command = stress_string.format(fault_command, fault_config)
        print("[x] Stress command to run: " + shell_command)
        self.fault_injection_start_times_ms.append(utils.current_milli_time())
        self.fault_injection_process = subprocess.Popen(shell_command, shell=True)
        return pb2.EmptyProto()

    def stop_fault_injection(self, request, context):
        try:
            os.kill(self.fault_injection_process.pid, signal.SIGTERM)
            self.fault_injection_stop_times_ms.append(utils.current_milli_time())
            print("[x] Fault Injection Process Killed.")
        except:
            pass
        return pb2.EmptyProto()

    def inject_fault_after_delay(self, request, context):
        print("[x] Request received on inject fault after delay")
        self.fault_injection_parent_process = threading.Thread(target=self.run_command,
                                                               args=(request.delay, request.fault_command,
                                                                     request.fault_config))
        # self.fault_injection_parent_process.start()
        print("[x] Responding to the user request")
        return pb2.EmptyProto()

    def run_command(self, delay, fault_command, fault_config):
        time.sleep(delay)
        fault_command = fault_command
        fault_config = fault_config
        stress_string = 'stress-ng {0} {1}'
        shell_command = stress_string.format(fault_command, fault_config)
        print("[x] Starting Fault Injection")
        self.fault_injection_start_times_ms.append(utils.current_milli_time())
        self.fault_injection_process = subprocess.Popen(shell_command, shell=True)

    def get_resource_logs(self, request, context):
        print("[x] Received resource log request.")

        # --- stop power measurement thread ---
        try:
            if self.power_thread:
                self.power_thread.stop()
                self.power_thread.join()
                print("[x] Power thread stopped.")
        except Exception as e:
            print(f"[!] Error stopping power thread: {e}")

        # --- stop resource utilization thread ---
        try:
            if self.resource_thread:
                self.resource_thread.stop()
                self.resource_thread.join()
                print("[x] Resource thread stopped.")
        except Exception as e:
            print(f"[!] Error stopping resource thread: {e}")

        # --- stop fault injection if still alive ---
        try:
            if self.fault_injection_parent_process:
                os.kill(self.fault_injection_parent_process.pid, signal.SIGTERM)
                print("[x] Fault Injection Parent Process Killed.")
        except Exception:
            pass
        try:
            if self.fault_injection_process:
                os.kill(self.fault_injection_process.pid, signal.SIGTERM)
                print("[x] Fault Injection Process Killed.")
        except Exception:
            pass

        self.fault_injection_stop_times_ms.append(utils.current_milli_time())

        # --- read resource logs ---
        with open('cpu_utilization.log', "rb") as f:
            cpu_data = f.read()
        with open('memory_utilization.log', "rb") as f:
            memory_data = f.read()
        with open('network_utilization.log', "rb") as f:
            network_data = f.read()
        with open('ios_utilization.log', "rb") as f:
            ios_data = f.read()

        cpu_file_data = pb2.FileData(data=cpu_data)
        memory_file_data = pb2.FileData(data=memory_data)
        network_file_data = pb2.FileData(data=network_data)
        ios_file_data = pb2.FileData(data=ios_data)

        # --- collect temperatures ---
        temp_timestamps, cpu_temperatures = [], []
        try:
            if self.power_thread:
                temp_timestamps, cpu_temperatures = self.power_thread.get_temperatures()
        except Exception as e:
            print(f"[!] Error collecting temperatures: {e}")

        resource_logs = pb2.ResourceLogs(
            cpu_log=cpu_file_data,
            memory_log=memory_file_data,
            io_log=ios_file_data,
            network_log=network_file_data,
            fault_injection_start_times_ms=self.fault_injection_start_times_ms,
            fault_injection_stop_times_ms=self.fault_injection_stop_times_ms,
            temperature_timestamps_ms=temp_timestamps,
            cpu_temperatures=cpu_temperatures,
        )

        # reset fault injection state
        self.fault_injection_start_times_ms = []
        self.fault_injection_stop_times_ms = []

        return resource_logs

    # OLD CODE BELOW

    def get_cpu_trace(self, request, context):
        # This method gives the cpu load over the last 1 minute, 5 minutes, and 15 minutes
        # Since our experiments duration is 1 minute --> we can use the first one
        load1, load5, load15 = psutil.getloadavg()
        cpu_trace = pb2.CPUTrace()
        cpu_trace.cpu_load = (load1 / psutil.cpu_count()) * 100
        return cpu_trace

    def get_memory_usage(self, request, context):
        file = open("cpu.txt", "r")
        sum = 0
        count = 0
        for line in file:
            cpu = float(line.strip())
            sum += cpu
            count += 1
        file.close()
        memory_file = open("memory.txt", "r")
        memory_sum = 0
        memory_count = 0
        for line in memory_file:
            memory = float(line.strip())
            memory_sum += memory
            memory_count += 1
        memory_file.close()
        memory_trace = pb2.MemoryTrace()
        memory_trace.current_memory_mb = sum / count
        memory_trace.peak_memory_mb = memory_sum / memory_count
        return memory_trace

    def start_memory_tracing(self, request, context):
        print("[x] Tracing resource utilization. ")
        cpu_tracing_process = subprocess.Popen("top -b -d 1 -n 60 | grep 'Cpu(s)' | awk '{print $2}' > cpu.txt",
                                               shell=True)
        memory_tracing_process = subprocess.Popen(
            "top -b -d 1 -n 60 | grep 'MiB Mem :   7808.0 total,' | awk '{print $8/7808.0 * 100}' > memory.txt",
            shell=True)
        return pb2.EmptyProto()


import glob

def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None

def _is_raspberry_pi():
    # Most reliable: device-tree model string contains 'Raspberry Pi'
    model = _read_text("/sys/firmware/devicetree/base/model") or _read_text("/proc/device-tree/model")
    return bool(model and "raspberry pi" in model.lower())

def _is_jetson():
    model = _read_text("/sys/firmware/devicetree/base/model") or _read_text("/proc/device-tree/model")
    if model and "jetson" in model.lower():
        return True
    # Fallback: tegrastats exists
    return os.path.exists("/usr/bin/tegrastats") or os.path.exists("/bin/tegrastats")

# def _read_rpi_cpu_temp():
#     """
#     Returns CPU temp in °C using vcgencmd; None if unavailable.
#     """
#     # try:
#     #     # vcgencmd typically resides in /usr/bin on Raspberry Pi OS
#     #     proc = subprocess.Popen(["/usr/bin/vcgencmd", "measure_temp"],
#     #                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     #     out, _ = proc.communicate(timeout=2)
#     #     # Example output: b"temp=53.2'C\n"
#     #     s = out.decode(errors="ignore").strip()
#     #     if "temp=" in s:
#     #         val = s.split("temp=")[1].split("'")[0]
#     #         return float(val), proc
#     # except Exception:
#     #     pass
#     # return None, None
#     try:
#         with open("/sys/class/thermal/thermal_zone0/temp") as f:
#             return float(f.read()) / 1000.0, None
#     except Exception:
#         return None, None
#
# def _read_jetson_cpu_temp_from_sysfs():
#     """
#     Read CPU temperature from thermal zones on Jetson.
#     Returns (temp_c, None) or (None, None).
#     """
#     # Search thermal zones for CPU-like sensors
#     # Typical types: CPU-therm, Tdiode, GPU-therm, etc.
#     try:
#         for tz in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*")):
#             ttype = _read_text(os.path.join(tz, "type"))
#             if not ttype:
#                 continue
#             if "cpu" in ttype.lower() or ttype.lower() in {"cpu-therm", "cpu_therm"}:
#                 tval = _read_text(os.path.join(tz, "temp"))
#                 if tval and tval.isdigit():
#                     # millidegree C
#                     return (int(tval) / 1000.0, None)
#                 # Some boards may expose raw °C as float/int
#                 try:
#                     return (float(tval), None)
#                 except Exception:
#                     pass
#         # Fallback: some Jetsons expose this file
#         alt = _read_text("/sys/devices/gpu.0/temp")
#         if alt:
#             try:
#                 v = int(alt) / 1000.0 if alt.isdigit() else float(alt)
#                 return (v, None)
#             except Exception:
#                 pass
#     except Exception:
#         pass
#     return None, None
#
# def _read_jetson_cpu_temp_from_tegrastats():
#     """
#     Parse tegrastats one-shot output.
#     Returns (temp_c, proc_or_none).
#     """
#     try:
#         # Run once; tegrastats prints a line and keeps running, so use timeout+kill
#         proc = subprocess.Popen(["tegrastats", "--interval", "1000"],
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         # Read a single line with a short timeout
#         out = b""
#         start = time.time()
#         while time.time() - start < 2:
#             chunk = proc.stdout.readline()
#             if not chunk:
#                 break
#             out += chunk
#             if b"\n" in chunk:
#                 break
#         # Try to stop it cleanly
#         try:
#             proc.terminate()
#         except Exception:
#             pass
#         # Example snippet often contains: "CPU@52.5C GPU@... AO@..."
#         line = out.decode(errors="ignore")
#         for token in line.replace("@", " ").split():
#             if token.endswith("C"):
#                 # Take first numeric preceding 'C' (likely CPU)
#                 try:
#                     # tokens like 'CPU', '52.5C' -> we search numeric
#                     val = token[:-1]
#                     return (float(val), proc)
#                 except Exception:
#                     continue
#     except Exception:
#         pass
#     return None, None


class PowerMeasurementThread(threading.Thread):
    """
    Cross-device temperature sampler (Raspberry Pi & Jetson Nano).
    Avoids repeated fork() calls that trigger gRPC warnings.
    """

    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.stop_flag = False
        self.timestamps = []
        self.temps = []
        self._tegrastats_proc = None

        # Detect platform once
        self._on_rpi = _is_raspberry_pi()
        self._on_jetson = (not self._on_rpi) and _is_jetson()

    def run(self):
        if self._on_jetson and not self._have_sysfs_temp():
            # Start one long-running tegrastats reader
            self._tegrastats_proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval * 1000)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            )

        while not self.stop_flag:
            ts = utils.current_milli_time()
            temp_c = None

            # --- Raspberry Pi: use sysfs
            if self._on_rpi:
                temp_c = self._read_sysfs_temp("/sys/class/thermal/thermal_zone0/temp")

            # --- Jetson: try sysfs first
            if self._on_jetson and temp_c is None:
                temp_c = self._read_jetson_sysfs()

            # --- Jetson fallback: read from tegrastats process
            if self._on_jetson and temp_c is None and self._tegrastats_proc:
                try:
                    line = self._tegrastats_proc.stdout.readline()
                    if line:
                        for token in line.replace("@", " ").split():
                            if token.endswith("C"):
                                temp_c = float(token[:-1])
                                break
                except Exception:
                    pass

            # --- Generic Linux fallback
            if temp_c is None:
                temp_c = self._read_hwmon_generic()

            if temp_c is not None:
                self.timestamps.append(ts)
                self.temps.append(temp_c)

            time.sleep(self.interval)

    def stop(self):
        self.stop_flag = True
        if self._tegrastats_proc:
            try:
                self._tegrastats_proc.terminate()
            except Exception:
                pass

    # ------------------ helpers ------------------

    def _have_sysfs_temp(self):
        return bool(glob.glob("/sys/devices/virtual/thermal/thermal_zone*"))

    def _read_sysfs_temp(self, path):
        try:
            with open(path) as f:
                raw = f.read().strip()
                return int(raw) / 1000.0 if raw.isdigit() else float(raw)
        except Exception:
            return None

    def _read_jetson_sysfs(self):
        try:
            for tz in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
                ttype = _read_text(os.path.join(tz, "type"))
                if ttype and "cpu" in ttype.lower():
                    val = _read_text(os.path.join(tz, "temp"))
                    if val and val.isdigit():
                        return int(val) / 1000.0
                    try:
                        return float(val)
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    def _read_hwmon_generic(self):
        try:
            for p in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
                raw = _read_text(p)
                if raw and raw.isdigit():
                    v = int(raw) / 1000.0
                    if 0.0 < v < 120.0:
                        return v
        except Exception:
            pass
        return None

    # ------------------ public API ------------------

    def get_temperatures(self):
        return self.timestamps, self.temps

    def get_average_power(self):
        return (sum(self.temps) / len(self.temps)) if self.temps else 0.0

class ResourceUtilizationThread(threading.Thread):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.stop_flag = False
        self.cpu_data = []
        self.memory_data = []
        self.disk_data = []
        self.network_received_speed = []
        self.network_transmitted_speed = []

    def run(self):
        # Start the sar command to collect CPU, memory, and network utilization data
        cpu_process = subprocess.Popen("sar -u 1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        memory_process = subprocess.Popen("sar -r 1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        network_process = subprocess.Popen("sar -n DEV 1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Start the iostat command to collect disk utilization data
        iostat_command = f"iostat -dkx {self.interval}"
        iostat_process = subprocess.Popen(iostat_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while not self.stop_flag:
            cpu_line = cpu_process.stdout.readline()
            memory_line = memory_process.stdout.readline()
            network_line = network_process.stdout.readline()
            iostat_line = iostat_process.stdout.readline()

            if cpu_line.strip() and not (cpu_line.startswith(b"Linux") or cpu_line.startswith(b"Average")):
                cpu_fields = cpu_line.split()
                if len(cpu_fields) == 9 and cpu_fields[8] != b'%idle':
                    self.cpu_data.append(100.0 - float(cpu_fields[8]))

            if memory_line.strip() and not (memory_line.startswith(b"Linux") or memory_line.startswith(b"Average")):
                mem_fields = memory_line.split()
                if len(mem_fields) > 5 and mem_fields[5] != b'%memused':
                    self.memory_data.append(float(mem_fields[5]))

            if network_line.strip() and not (network_line.startswith(b"Linux") or network_line.startswith(b"Average")):
                net_fields = network_line.split()
                if len(net_fields) > 4 and net_fields[2] == b'wlan0':
                    self.network_received_speed.append(float(net_fields[5]))
                    self.network_transmitted_speed.append(float(net_fields[6]))

            if iostat_line.strip() and not (iostat_line.startswith(b"Linux") or iostat_line.startswith(b"Device")):
                iostat_fields = iostat_line.split()
                if len(iostat_fields) > 22 and iostat_fields[0] == b'mmcblk0':
                    self.disk_data.append(float(iostat_fields[22]))

        # Terminate the sar and iostat processes after the loop is done
        cpu_process.terminate()
        memory_process.terminate()
        network_process.terminate()
        iostat_process.terminate()

    def stop(self):
        self.stop_flag = True

    def get_average_cpu_utilization(self):
        return sum(self.cpu_data) / len(self.cpu_data)

    def get_average_memory_utilization(self):
        return sum(self.memory_data) / len(self.memory_data)

    def get_average_disk_utilization(self):
        return sum(self.disk_data) / len(self.disk_data)

    def get_average_network_utilization(self):
        return sum(self.network_received_speed) / len(self.network_received_speed), sum(
            self.network_transmitted_speed) / len(self.network_transmitted_speed)


class ResourceUtilizationSavingThread(threading.Thread):
    def __init__(self, interval, timeout):
        super().__init__()
        self.interval = interval
        self.timeout = timeout
        self.cpu_process = None
        self.memory_process = None
        self.network_process = None
        self.iostat_process = None

    def run(self):
        # Delete previous files:
        subprocess.run("rm cpu_utilization.log", shell=True)
        subprocess.run("rm memory_utilization.log", shell=True)
        subprocess.run("rm network_utilization.log", shell=True)
        subprocess.run("rm ios_utilization.log", shell=True)
        # Start the sar command to collect CPU, memory, and network utilization data
        cpu_command = f"sar -u ALL {self.interval} {self.timeout}> cpu_utilization.log"
        self.cpu_process = subprocess.Popen(cpu_command, shell=True)
        self.memory_process = subprocess.Popen(f"sar -r ALL {self.interval} {self.timeout} > memory_utilization.log", shell=True)
        self.network_process = subprocess.Popen(f"sar -n DEV {self.interval} {self.timeout} > network_utilization.log ", shell=True)
        self.iostat_process = subprocess.Popen(f"iostat -t mmcblk0 -dkx {self.interval} {self.timeout} > ios_utilization.log",
                                               shell=True)

    def stop(self):
        self.cpu_process.terminate()
        self.memory_process.terminate()
        self.network_process.terminate()
        self.iostat_process.terminate()
        print("Terminated CPU SAR Process")
        print("Terminated MEM SAR Process")
        print("Terminated NET SAR Process")
        print("Terminated IOSTAT Process")

    def get_cpu_process(self):
        return self.cpu_process

    def get_memory_process(self):
        return self.memory_process

    def get_network_process(self):
        return self.network_process

    def get_io_process(self):
        return self.iostat_process
