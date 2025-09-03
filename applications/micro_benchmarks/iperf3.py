import os
import signal
import subprocess
import time as t
from protos import benchmark_pb2 as pb2
from utils import current_milli_time


def run_iperf(request, request_received_time_ms):
    try:
        iCommand = 'iperf3 -c 192.168.0.159 -t 1 -P 4 -i 1 -J'
        subprocess.check_output(iCommand, shell=True).decode('UTF-8')
        print("Packeges from the client has been sent to the server.\n")

    except Exception as e:
        print("Following exception has occured:", e)
        print("\nExiting the program.\n")
        exit(0)
    iperf_response = pb2.IperfResult()
    iperf_response.request_time_ms = request.request_time_ms
    iperf_response.request_received_time_ms = request_received_time_ms
    iperf_response.response_time_ms = current_milli_time()

    return iperf_response
