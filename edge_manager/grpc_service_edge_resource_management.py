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

    def __init__(self, *args, **kwargs):
        self.resource_thread = None
        self.power_thread = None

        self.fault_injection_process = None
        self.fault_injection_parent_process = None
        self.fault_injection_start_times_ms = []
        self.fault_injection_stop_times_ms = []
        self.utilization_output = None
        self.resource_tracing_process = None

    def start_resource_tracing(self, request, context):
        print("[x] Tracing resource utilization. ")
        # Start the resource utilization thread
        self.resource_thread = ResourceUtilizationThread(interval=1)
        self.resource_thread.start()
        # Start the power measurement thread
        self.power_thread = PowerMeasurementThread(interval=1)
        self.power_thread.start()
        return pb2.EmptyProto()

    def start_resource_tracing_and_saving(self, request, context):
        print("[x] Tracing resource utilization. ")
        # Start the resource utilization thread
        self.resource_thread = ResourceUtilizationSavingThread(interval=1, timeout=request.timeout)
        self.resource_thread.start()
        # Start the power measurement thread
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
        try:
            os.kill(self.power_thread.get_power_process().pid, signal.SIGTERM)
        except:
            pass
        self.power_thread.stop()
        self.power_thread.join()
        print("[x] Power thread stopped.")
        # Stop the resource utilization thread and get the average values
        self.resource_thread.stop()
        self.resource_thread.join()
        print("[x] Resource thread stopped.")
        # Send a signal to the stress process to terminate it
        try:
            os.kill(self.fault_injection_parent_process.pid, signal.SIGTERM)
            print("[x] Fault Injection Parent Process Killed.")
        except:
            pass
        try:
            os.kill(self.fault_injection_process.pid, signal.SIGTERM)
            print("[x] Fault Injection Process Killed.")
        except:
            pass
        self.fault_injection_stop_times_ms.append(utils.current_milli_time())

        with open('cpu_utilization.log', "rb") as f:
            cpu_data = f.read()

        with open('memory_utilization.log', "rb") as f:
            memory_data = f.read()

        with open('network_utilization.log', "rb") as f:
            network_data = f.read()

        with open('ios_utilization.log', "rb") as f:
            ios_data = f.read()

        cpu_file_data = pb2.FileData()
        cpu_file_data.data = cpu_data

        memory_file_data = pb2.FileData()
        memory_file_data.data = memory_data

        network_file_data = pb2.FileData()
        network_file_data.data = network_data

        ios_file_data = pb2.FileData()
        ios_file_data.data = ios_data

        temperature_timestamps_ms, cpu_temperatures = self.power_thread.get_temperatures()

        resource_logs = pb2.ResourceLogs(cpu_log=cpu_file_data, memory_log=memory_file_data, io_log=ios_file_data,
                                         network_log=network_file_data,
                                         fault_injection_start_times_ms=self.fault_injection_start_times_ms,
                                         fault_injection_stop_times_ms=self.fault_injection_stop_times_ms,
                                         temperature_timestamps_ms=temperature_timestamps_ms,
                                         cpu_temperatures=cpu_temperatures)
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


class PowerMeasurementThread(threading.Thread):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.stop_flag = False
        self.power_timestamps = []
        self.power_data = []
        self.power_process = None

    def run(self):
        while not self.stop_flag:
            # Run vcgencmd to get power consumption data
            command = "sudo vcgencmd measure_temp"
            self.power_timestamps.append(utils.current_milli_time())
            self.power_process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = self.power_process.communicate()

            # Parse the output to extract the voltage and current values
            temperature = float(output.decode().split("=")[1][:-3])

            # Add the power consumption data to the list
            self.power_data.append(temperature)

            # Wait for the measurement interval
            time.sleep(self.interval)

    def get_power_process(self):
        return self.power_process

    def stop(self):
        self.stop_flag = True

    def get_temperatures(self):
        return self.power_timestamps, self.power_data

    def get_average_power(self):
        return sum(self.power_data) / len(self.power_data)


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
