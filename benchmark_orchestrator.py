import pyshark
import time
import asyncio
import threading
import csv

import configs
import utils
import grpc_client
import benchmark_orchestrator

wireshark_thread = None
wireshark_output = None
saved_wireshark_capture = True
packet_list = []


####################################################################################
####################################################################################
## Capturing packets with wireshark --> Pyshark Library
####################################################################################
####################################################################################
def capture_packets_with_wireshark(application, cpu_cores, experiment_id):
    while benchmark_orchestrator.saved_wireshark_capture is not True:
        print("[!!] Not ready to start new experiment, waiting for wireshark capture")
        time.sleep(3)  # Just to avoid hogging the CPU
    else:
        print("********[x]***** Wireshark capture is ready for new experiment.")
    benchmark_orchestrator.wireshark_filename = "./results/wireshark/" + application + "/" \
                                                + configs.EDGE_DEVICE_NAME + "-cpuStress" + \
                                                str(cpu_cores) + "-exp" + str(experiment_id) + ".pcap"

    # ************** Starting wireshark capture *************** #
    benchmark_orchestrator.wireshark_output = open(benchmark_orchestrator.wireshark_filename, "w")
    print("********[x]***** Openned wireshark output")
    capture = pyshark.LiveCapture(interface="en0", output_file=benchmark_orchestrator.wireshark_filename)
    print("********[x]***** Live Capture")
    benchmark_orchestrator.wireshark_thread = threading.Thread(target=f, args=(capture,))
    benchmark_orchestrator.wireshark_thread.setDaemon(True)
    benchmark_orchestrator.wireshark_thread.start()


def process_packets(packet):
    global packet_list
    try:
        packet_version = None
        layer_name = None
        if len(packet.layers) > 0:
            packet_version = packet.layers[1].version
        if len(packet.layers) > 1:
            layer_name = packet.layers[2].layer_name
        packet_list.append(f'{packet_version}, {layer_name}, {packet.length}')
    except AttributeError:
        pass


def f(capture):
    print("********[x]***** Starting to sniff with timeout: " + str(configs.MAX_EXPERIMENT_TIME_SECONDS))
    benchmark_orchestrator.saved_wireshark_capture = False
    try:
        capture.apply_on_packets(process_packets, timeout=configs.MAX_EXPERIMENT_TIME_SECONDS)
    except asyncio.TimeoutError:
        print("********[x]***** Sniffing done with timed out")
    # capture.sniff(timeout=max_time)
    print("********[x]***** Sniffing is done")
    capture.close()
    benchmark_orchestrator.wireshark_output.close()
    benchmark_orchestrator.saved_wireshark_capture = True
    print("********[x]***** Sniff Result Saved")


####################################################################################
####################################################################################
## Saving latency and resource utilization results in csv file
####################################################################################
####################################################################################
def save_experiment_results(client, application, cpu_cores, experiment_id, results):
    latency_filename = "./results/latency/" + application + "/" + configs.EDGE_DEVICE_NAME + "-cpuStress" + \
                       str(cpu_cores) + "-exp" + str(experiment_id) + ".csv"
    print("********[x]***** Saving results for filename:{}".format(latency_filename))
    print("********[x]***** Size of results: " + str(len(results)))
    with open(latency_filename, 'w', encoding='UTF8', newline='') as csv_output:
        # create the csv writer
        writer = csv.writer(csv_output)

        header = ['frame_id', 'request_time_ms', 'request_received_time_ms', 'response_time_ms',
                  'response_received_time_ms', 'end_to_end_latency', 'compute_time', 'transmission_time']
        writer.writerow(header)
        for result in results:
            end_to_end_latency = result.response_received_time_ms - result.request_time_ms
            compute_time = result.response_time_ms - result.request_received_time_ms
            transmission_time = end_to_end_latency - compute_time
            row = [result.frame_id, result.request_time_ms, result.request_received_time_ms,
                   result.response_time_ms, result.response_received_time_ms, end_to_end_latency,
                   compute_time, transmission_time]
            writer.writerow(row)

        writer.writerow(['CPU_usage_percent', 'peak_memory_mb', 'current_memory_mb'])
        cpu_trace = client.call_server_for_cpu_trace()
        memory_trace = client.call_server_for_memory_trace()
        writer.writerow([cpu_trace.cpu_load, memory_trace.peak_memory_mb, memory_trace.current_memory_mb])


####################################################################################
####################################################################################
####################################################################################
####################################################################################
def run_single_experiment(client, application, cpu_cores_to_stress, experiment_id):
    experiment_results = []
    capture_packets_with_wireshark(application, cpu_cores_to_stress, experiment_id)
    client.call_server_to_start_mem_tracing()
    client.call_server_to_stress_cpu(cpu_cores_to_stress, configs.MAX_EXPERIMENT_TIME_SECONDS)

    # **** Starting the experiment ****************** #
    frame_id = 1
    max_frame = configs.MAX_FRAME_NUM
    while frame_id <= max_frame:
        # **** Read image frame ********************* #
        input_image = utils.read_input_workload_frame(frame_id)
        start_time = time.time()
        # **** Make gRPC call based on the application **** #
        if application == 'object_tracking':
            grpc_result = client.call_object_tracking_server(image=input_image, frame_id=frame_id)
        else:
            grpc_result = client.call_object_detection_server(image=input_image, frame_id=frame_id)
        response_received_time_ms = utils.current_milli_time()
        grpc_result.response_received_time_ms = response_received_time_ms
        experiment_results.append(grpc_result)

        # **** BEGIN: Following the experiment's FPS config with sleep **** #
        end_time = time.time()
        processing_time = end_time - start_time
        if processing_time < 1. / configs.FPS:
            wait_time = (1. / configs.FPS) - processing_time
            time.sleep(wait_time)
        start_time = end_time
        # **** END: Following the experiment's FPS config with sleep ****** #

        frame_id += 1

    print("********[x]***** Stopping the wireshark thread")
    benchmark_orchestrator.wireshark_thread.join()
    print("********[x]***** Saving experiment results")
    save_experiment_results(client, application, cpu_cores_to_stress, experiment_id, experiment_results)


if __name__ == '__main__':
    client = grpc_client.Client()

    for application in configs.APPLICATIONS:
        cpu_cores_to_stress = 0
        while cpu_cores_to_stress <= configs.MAX_CPU_CORES_TO_STRESS:

            experiment_id = 1
            while experiment_id <= configs.REPEAT_EXPERIMENTS:
                run_single_experiment(client, application, cpu_cores_to_stress, experiment_id)
                experiment_id += 1

            cpu_cores_to_stress += 1