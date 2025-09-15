# import sys
# sys.path.append('..')
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import grpc
from concurrent import futures
import configs
from protos import benchmark_pb2_grpc as pb2_grpc
from grpc_service_micro_benchmarks import MicroBenchmarksGRPCService
from grpc_service_application_benchmarks import ApplicationBenchmarksGRPCService
from grpc_service_edge_resource_management import EdgeResourceManagementGRPCService


def serve():
    # Set the maximum message size to 100 MB (100 * 1024 * 1024 bytes)
    max_size = 100 * 1024 * 1024

    # Create the server options with the maximum message size
    options = [
        ('grpc.max_receive_message_length', max_size),
        ('grpc.max_send_message_length', max_size)
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    pb2_grpc.add_ApplicationBenchmarksServicer_to_server(ApplicationBenchmarksGRPCService(), server)
    pb2_grpc.add_MicroBenchmarksServicer_to_server(MicroBenchmarksGRPCService(), server)
    pb2_grpc.add_EdgeResourceManagementServicer_to_server(EdgeResourceManagementGRPCService(), server)

    server.add_insecure_port('[::]:{0}'.format(configs.EDGE_DEVICE_PORT))
    print("Listening on Port {0} ...".format(configs.EDGE_DEVICE_PORT))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
