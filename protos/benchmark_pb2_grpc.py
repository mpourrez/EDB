# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from protos import benchmark_pb2 as benchmark__pb2


class BenchmarksStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.track_objects = channel.unary_unary(
                '/protos.Benchmarks/track_objects',
                request_serializer=benchmark__pb2.Request.SerializeToString,
                response_deserializer=benchmark__pb2.DetectionTrackingResponse.FromString,
                )
        self.detect_objects = channel.unary_unary(
                '/protos.Benchmarks/detect_objects',
                request_serializer=benchmark__pb2.Request.SerializeToString,
                response_deserializer=benchmark__pb2.DetectionTrackingResponse.FromString,
                )
        self.start_memory_tracing = channel.unary_unary(
                '/protos.Benchmarks/start_memory_tracing',
                request_serializer=benchmark__pb2.EmptyProto.SerializeToString,
                response_deserializer=benchmark__pb2.EmptyProto.FromString,
                )
        self.get_cpu_trace = channel.unary_unary(
                '/protos.Benchmarks/get_cpu_trace',
                request_serializer=benchmark__pb2.EmptyProto.SerializeToString,
                response_deserializer=benchmark__pb2.CPUTrace.FromString,
                )
        self.get_memory_usage = channel.unary_unary(
                '/protos.Benchmarks/get_memory_usage',
                request_serializer=benchmark__pb2.EmptyProto.SerializeToString,
                response_deserializer=benchmark__pb2.MemoryTrace.FromString,
                )


class BenchmarksServicer(object):
    """Missing associated documentation comment in .proto file."""

    def track_objects(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def detect_objects(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def start_memory_tracing(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_cpu_trace(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_memory_usage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BenchmarksServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'track_objects': grpc.unary_unary_rpc_method_handler(
                    servicer.track_objects,
                    request_deserializer=benchmark__pb2.Request.FromString,
                    response_serializer=benchmark__pb2.DetectionTrackingResponse.SerializeToString,
            ),
            'detect_objects': grpc.unary_unary_rpc_method_handler(
                    servicer.detect_objects,
                    request_deserializer=benchmark__pb2.Request.FromString,
                    response_serializer=benchmark__pb2.DetectionTrackingResponse.SerializeToString,
            ),
            'start_memory_tracing': grpc.unary_unary_rpc_method_handler(
                    servicer.start_memory_tracing,
                    request_deserializer=benchmark__pb2.EmptyProto.FromString,
                    response_serializer=benchmark__pb2.EmptyProto.SerializeToString,
            ),
            'get_cpu_trace': grpc.unary_unary_rpc_method_handler(
                    servicer.get_cpu_trace,
                    request_deserializer=benchmark__pb2.EmptyProto.FromString,
                    response_serializer=benchmark__pb2.CPUTrace.SerializeToString,
            ),
            'get_memory_usage': grpc.unary_unary_rpc_method_handler(
                    servicer.get_memory_usage,
                    request_deserializer=benchmark__pb2.EmptyProto.FromString,
                    response_serializer=benchmark__pb2.MemoryTrace.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'protos.Benchmarks', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Benchmarks(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def track_objects(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.Benchmarks/track_objects',
            benchmark__pb2.Request.SerializeToString,
            benchmark__pb2.DetectionTrackingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def detect_objects(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.Benchmarks/detect_objects',
            benchmark__pb2.Request.SerializeToString,
            benchmark__pb2.DetectionTrackingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def start_memory_tracing(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.Benchmarks/start_memory_tracing',
            benchmark__pb2.EmptyProto.SerializeToString,
            benchmark__pb2.EmptyProto.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_cpu_trace(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.Benchmarks/get_cpu_trace',
            benchmark__pb2.EmptyProto.SerializeToString,
            benchmark__pb2.CPUTrace.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_memory_usage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.Benchmarks/get_memory_usage',
            benchmark__pb2.EmptyProto.SerializeToString,
            benchmark__pb2.MemoryTrace.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
