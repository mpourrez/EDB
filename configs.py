#!/usr/bin/env python
# encoding: utf-8
"""
configs.py
"""

from enum import Enum
class EdgeDevice(Enum):
    RPI = 'raspberrypi'
    NANO = 'nano'
    CORAL = 'coral'

EDGE_DEVICE_NAME = EdgeDevice.RPI
if EDGE_DEVICE_NAME == EdgeDevice.RPI:
    EDGE_DEVICES_IP = ['192.168.0.139']
    # EDGE_DEVICES_IP = ['192.168.0.151', '192.168.0.194', '192.168.0.244']
    PROJECT_PATH = "/home/pi/Projects/"
else:
    EDGE_DEVICES_IP = ['192.168.0.225']
    # EDGE_DEVICES_IP = ['192.168.0.122', '192.168.0.225','192.168.0.141']
    PROJECT_PATH = "/home/nano/Projects/"
EDGE_DEVICE_PORT = 50051
ORCHESTRATOR_IP = '192.168.0.120'

MAX_FRAME_NUM = 300
PROJECT_PATH = "/Users/maryampourreza/Projects/"
WORKLOAD_INPUT_PATH = 'workloads/MOT20-01/img1/'
TIME_BOUND_FOR_FAULT_INJECTION = 5  # in-seconds


# Groups of replicas by device type
DEVICE_TYPE = "pi"
# PI_GROUP = [
#     ("192.168.0.168", "pi1"),
#     ("192.168.0.12", "pi2"),
#     ("192.168.0.139", "pi3"),
# ]
# on seperate router
PI_GROUP = [
    ("192.168.0.151", "pi1"),
    ("192.168.0.194", "pi2"),
    ("192.168.0.244", "pi3"),
]

NANO_GROUP = [
    ("192.168.0.122", "nano1"),
    ("192.168.0.225", "nano2"),
    ("192.168.0.141", "nano3"),
]

# PI_GROUP = [
#     ("192.168.0.110", "pi1"),
#     ("192.168.0.104", "pi2"),
#     ("192.168.0.108", "pi3"),
# ]

# NANO_GROUP = [
#     ("192.168.0.113", "nano1"),
#     ("192.168.0.112", "nano2"),
#     ("192.168.0.111", "nano3"),
# ]

CHECKPOINT_PERIODS_SA = [5, 15, 30]  # seconds

REPLICATION_MODE = "PASSIVE"   # or "ACTIVE"
QUORUM_MODE = "FIRST"          # or "MAJORITY"

HEARTBEAT_MS = 1000
FAIL_DETECT_TIMEOUT_MS = 40000

REPETITIONS = 3
FAULT_FREE_DURATIONS = 60

SSH_USER = "pi"   # or "nano" depending
EDGE_MANAGER_CMD = "python3 /home/pi/Projects/EDB/edge_manager.py"

NUMBER_OF_FAULT_INJECTIONS = 1
NUMBER_OF_FAULT_FREE_ROUNDS = 1
FAULT_INJECTION_DURATION = 60
RESOURCE_MONITOR_INTERVALS = 1  # in-seconds
EXPERIMENT_DURATION = FAULT_INJECTION_DURATION * NUMBER_OF_FAULT_INJECTIONS + \
                      FAULT_FREE_DURATIONS * (NUMBER_OF_FAULT_INJECTIONS + 1)

TIMEOUT_THRESHOLDS = {
    # 'MM': [500, 600, 800],
    # 'PS': [2300],
    'FFT': [2300, 2800, 5500],
    'SORT': [17200, 18000, 19000],
    'IPERF': [10900, 11400, 12000],
    'IP': [700, 900, 5000],
    'IC-A-CPU': [3700, 4300, 5000],
    'OD-CPU': [1400, 2000, 3200],
    'OD-GPU': [600, 800, 1100],
    'PS': [5600, 5800, 6000],
    'SA-AGG': [5600, 5800, 6000],
}

# TIMEOUT_THRESHOLDS = {
#     'MM': [100000],
#     'FFT': [100000],
#     'SORT': [10000],
#     'IPERF': [10000],
#     'IP': [10000],
#     'IC-A-CPU': [10000],
#     'OD-CPU': [10000],
# }

# APPLICATIONS = ['IPERF', 'IP', 'OD-CPU', 'PS', 'OD-GPU']
APPLICATIONS = ['IP']
# APPLICATIONS = ['FFT', 'SORT', 'IP', 'OD-CPU', 'OD-GPU', 'SA-AGG']
# APPLICATIONS = ['OD-CPU', 'OD-GPU', 'SA-AGG', 'FFT']
# ,  'OD-CPU', 'OD-GPU', 'FFT']
# APPLICATIONS = ['OD-CPU', 'OD-GPU', 'SA-AGG']
# APPLICATIONS = ['FFT', 'IP', 'PS', 'OD-CPU', 'OD-GPU']
# APPLICATIONS = ['FFT', 'SORT', 'IPERF', 'IP', 'PS', 'OD-CPU', 'SA-AGG', 'OD-GPU']
#
# APPLICATIONS = ['SORT', 'DD', 'IPERF', 'IP', 'SA', 'PS', 'AE', 'OD-CPU']

# APPLICATIONS = ['MM', 'FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
#                  'IP', 'SA', 'ST', 'IC-A-CPU', 'IC-S-CPU', 'OD-CPU', 'PS', 'AE', 'OT-CPU']
CURRENT_APP = "FFT"
CURRENT_FAULT = "CPU"

class Fault:
    def __init__(self, fault_name, abbreviation, fault_command, fault_config):
        self.fault_name = fault_name
        self.abbreviation = abbreviation
        self.fault_command = fault_command
        self.fault_config = fault_config

FAULTS = [
    # Fault('cpu-overload', 'CPU', '--cpu 0 --cpu-load', ['90']),
    Fault('memory-contention', 'MEM', '--vm 0 --vm-method all --vm-bytes', ['60%']),
    # Fault('io-stress', 'IO', '--io', ['100']),
    # Fault('cache-thrashing', 'CCHE', '--cache', ['0']),
    # Fault('page-fault', 'PF', '--fault', ['0']),

    # Fault('cpu-overload', 'CPU', '--cpu 0 --cpu-load', ['20', '60', '90']),
    # Fault('memory-contention', 'MEM', '--vm 0 --vm-method all --vm-bytes', ['20%', '60%', '90%']),
    # Fault('io-stress', 'IO', '--io', ['100']),
    # Fault('page-fault', 'PF', '--fault', ['0']),
    # Fault('cache-thrashing', 'CCHE', '--cache', ['0']),
    # Fault('context-switch', 'CTXS', '--cswitch --cswitch-ops', ['10000']),

    # Fault('memory-contention', 'MEM', '--vm 0 --vm-method all --vm-bytes', ['20%']),
    # Fault('cpu-overload', 'CPU', '--cpu 0 --cpu-load', ['80']),

    # Fault('cpu-overload', 'CPU', '--cpu 0 --cpu-load', ['20', '50', '80']),
    # # Fault('memory-contention', 'MEM', '--vm 0 --vm-method all --vm-bytes', ['20%', '60%', '90%']),
    # Fault('io-stress', 'IO', '--io', ['100']),
    # Fault('page-fault', 'PF', '--fault', ['0']),
    # Fault('cache-thrashing', 'CCHE', '--cache', ['0']),
    # Fault('context-switch', 'CTXS', '--cswitch --cswitch-ops', ['10000']),
    # Fault('interrupts', 'INTR', '--sleep ', ['32']),
    # # Fault('hdd-overload', 'HDD', '--hdd 0 --hdd-bytes', ['20%', '', '60%']),
    # Fault('ping-flood', 'TCP', '', ['u1000']),
    # Fault('ping-flood', 'PING', '', ['u1000']),
]

