from applications import sentiment_analysis, image_processing, speech_to_text, pocket_sphinx, aeneas, \
    object_detection_darknet_cpu, object_detection_darknet_gpu, object_tracker, stateful_sentiment_aggregator
from applications.image_classification import image_classification_alexnet_cpu, image_classification_alexnet_gpu, \
    image_classification_squeezenet_cpu, image_classification_squeezenet_gpu
from utils import *
from protos import benchmark_pb2_grpc as pb2_grpc

class ApplicationBenchmarksGRPCService(pb2_grpc.ApplicationBenchmarksServicer):

    def __init__(self, *args, **kwargs):
        self.fault_injection_process = None
        self.utilization_output = None
        self.resource_tracing_process = None

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
