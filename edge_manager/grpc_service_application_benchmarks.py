from applications import sentiment_analysis, image_processing, speech_to_text, pocket_sphinx, aeneas, \
    object_detection_darknet_cpu, object_detection_darknet_gpu, object_tracker, stateful_sentiment_aggregator, \
    slam_vf
from applications.image_classification import image_classification_alexnet_cpu, image_classification_alexnet_gpu, \
    image_classification_squeezenet_cpu, image_classification_squeezenet_gpu
from utils import *
from protos import benchmark_pb2_grpc as pb2_grpc

class ApplicationBenchmarksGRPCService(pb2_grpc.ApplicationBenchmarksServicer):

    def __init__(self, *args, **kwargs):
        self.fault_injection_process = None
        self.utilization_output = None
        self.resource_tracing_process = None
        # stateful_sentiment_aggregator.start_checkpoint_thread_if_needed()

    def image_processing(self, request, context):
        request_received_time_ms = current_milli_time()
        processing_result = image_processing.resize_image(request, request_received_time_ms)
        return processing_result

    def sentiment_analysis(self, request, context):
        request_received_time_ms = current_milli_time()
        analysis_result = sentiment_analysis.analyze_sentiment(request, request_received_time_ms)
        return analysis_result

    def speech_to_text(self, request, context):
        conversion_result = speech_to_text.convert_to_text(request)
        return conversion_result

    def image_classification_alexnet(self, request, context):
        request_received_time_ms = current_milli_time()
        classification_result = image_classification_alexnet_cpu.classify_image(request, request_received_time_ms)
        return classification_result

    def image_classification_alexnet_gpu(self, request, context):
        request_received_time_ms = current_milli_time()
        classification_result = image_classification_alexnet_gpu.classify_image(request, request_received_time_ms)
        return classification_result

    def image_classification_squeezenet(self, request, context):
        request_received_time_ms = current_milli_time()
        classification_result = image_classification_squeezenet_cpu.classify_image(request, request_received_time_ms)
        return classification_result

    def image_classification_squeezenet_gpu(self, request, context):
        request_received_time_ms = current_milli_time()
        classification_result = image_classification_squeezenet_gpu.classify_image(request, request_received_time_ms)
        return classification_result

    def object_detection_darknet(self, request, context):
        request_received_time_ms = current_milli_time()
        detection_result = object_detection_darknet_cpu.detect(request, request_received_time_ms)
        return detection_result

    def object_detection_darknet_gpu(self, request, context):
        request_received_time_ms = current_milli_time()
        detection_result = object_detection_darknet_gpu.detect(request, request_received_time_ms)
        return detection_result

    def pocket_sphinx(self, request, context):
        conversion_result = pocket_sphinx.convert_to_text(request)
        return conversion_result

    def aeneas(self, request, context):
        speech_to_text_result = aeneas.align_speech_text(request)
        return speech_to_text_result

    def object_tracking(self, request, context):
        request_received_time_ms = current_milli_time()
        tracking_result = object_tracker.track_from_image(request, request_received_time_ms)
        return tracking_result

    def object_tracking_gpu(self, request, context):
        request_received_time_ms = current_milli_time()
        object_tracker.enable_gpu()
        tracking_result = object_tracker.track_from_image(request, request_received_time_ms)
        return tracking_result

    def sentiment_aggregation(self, request, context):
        request_received_time_ms = current_milli_time()
        response = stateful_sentiment_aggregator.analyze_sentiment_stateful(request, request_received_time_ms)
        return response

    def slam_vf(self, request, context):
        request_received_time_ms = current_milli_time()
        return slam_vf.process_frame(request, request_received_time_ms)

    # ===== Replication-specific RPCs (multi-app router) =====
    # The (app, key) pair on each request selects which stateful app handles it.
    # Empty app falls back to "SA-AGG" for back-compat with legacy callers.
    @staticmethod
    def _resolve_app(req):
        return (getattr(req, "app", "") or "SA-AGG")

    def set_role_and_peers(self, request, context):
        # Apply to every stateful app on this edge; both pushers need to know
        # the current role/peers regardless of which app the orchestrator will
        # exercise next.
        ack = stateful_sentiment_aggregator.rpc_set_role_and_peers(request, context)
        try:
            slam_vf.set_role_and_peers(request.role, request.peer_hosts)
        except Exception as e:
            from protos import benchmark_pb2 as _pb2
            return _pb2.Ack(ok=False, msg=f"slam-vf role failed: {e}")
        return ack

    def set_checkpoint_period(self, request, context):
        app = self._resolve_app(request)
        if app == "SLAM-VF":
            slam_vf.set_checkpoint_period(request.seconds)
            from protos import benchmark_pb2 as _pb2
            return _pb2.Ack(ok=True, msg=f"slam-vf period={request.seconds}s")
        if app == "":
            # Empty app: legacy behavior — apply to SA-AGG only.
            return stateful_sentiment_aggregator.rpc_set_checkpoint_period(request, context)
        if app == "*":
            stateful_sentiment_aggregator.rpc_set_checkpoint_period(request, context)
            slam_vf.set_checkpoint_period(request.seconds)
            from protos import benchmark_pb2 as _pb2
            return _pb2.Ack(ok=True, msg=f"all stateful apps period={request.seconds}s")
        # Default: SA-AGG
        return stateful_sentiment_aggregator.rpc_set_checkpoint_period(request, context)

    def get_checkpoint(self, request, context):
        if self._resolve_app(request) == "SLAM-VF":
            return slam_vf.rpc_get_checkpoint(request, context)
        return stateful_sentiment_aggregator.rpc_get_checkpoint(request, context)

    def apply_checkpoint(self, request, context):
        if self._resolve_app(request) == "SLAM-VF":
            return slam_vf.rpc_apply_checkpoint(request, context)
        return stateful_sentiment_aggregator.rpc_apply_checkpoint(request, context)

    def get_current_version(self, request, context):
        if self._resolve_app(request) == "SLAM-VF":
            return slam_vf.rpc_get_current_version(request, context)
        return stateful_sentiment_aggregator.get_current_version(request, context)

    def get_log_tail(self, request, context):
        if self._resolve_app(request) == "SLAM-VF":
            return slam_vf.rpc_get_log_tail(request, context)
        return stateful_sentiment_aggregator.rpc_get_log_tail(request, context)

    def apply_log_tail(self, request, context):
        if self._resolve_app(request) == "SLAM-VF":
            return slam_vf.rpc_apply_log_tail(request, context)
        return stateful_sentiment_aggregator.rpc_apply_log_tail(request, context)

