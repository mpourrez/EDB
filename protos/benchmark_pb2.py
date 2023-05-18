# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: benchmark.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x62\x65nchmark.proto\x12\x06protos\"z\n\x1bMatrixMultiplicationRequest\x12 \n\x08matrix_1\x18\x01 \x01(\x0b\x32\x0e.protos.Matrix\x12 \n\x08matrix_2\x18\x02 \x01(\x0b\x32\x0e.protos.Matrix\x12\x17\n\x0frequest_time_ms\x18\x03 \x01(\x03\"\xb6\x01\n\x1cMatrixMultiplicationResponse\x12\x17\n\x0frequest_time_ms\x18\x01 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x02 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x03 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x04 \x01(\x03\x12\x1e\n\x06matrix\x18\x05 \x01(\x0b\x32\x0e.protos.Matrix\"R\n\x12\x46\x61stFourierRequest\x12#\n\x0einput_sequence\x18\x01 \x01(\x0b\x32\x0b.protos.Row\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xb2\x01\n\x13\x46\x61stFourierResponse\x12#\n\x0e\x66ourier_output\x18\x01 \x01(\x0b\x32\x0b.protos.Row\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"M\n\x14\x46loatingPointRequest\x12\x1c\n\x14\x66loating_point_input\x18\x01 \x01(\x03\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xae\x01\n\x15\x46loatingPointResponse\x12\x1d\n\x15\x66loating_point_output\x18\x01 \x01(\x02\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"L\n\x11\x46ileSorterRequest\x12\x1e\n\x04\x66ile\x18\x01 \x01(\x0b\x32\x10.protos.FileData\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xb3\x01\n\x12\x46ileSorterResponse\x12%\n\x0bsorted_file\x18\x01 \x01(\x0b\x32\x10.protos.FileData\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"$\n\tDDRequest\x12\x17\n\x0frequest_time_ms\x18\x01 \x01(\x03\"\x84\x01\n\nDDResponse\x12\x17\n\x0frequest_time_ms\x18\x01 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x02 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x03 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x04 \x01(\x03\"Y\n\x0cIperfRequest\x12\x10\n\x08hostname\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x05\x12\x10\n\x08\x64uration\x18\x03 \x01(\x05\x12\x17\n\x0frequest_time_ms\x18\x04 \x01(\x03\"\x98\x01\n\x0bIperfResult\x12\x11\n\tbandwidth\x18\x01 \x01(\x01\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"#\n\x06Matrix\x12\x19\n\x04rows\x18\x01 \x03(\x0b\x32\x0b.protos.Row\"\x15\n\x03Row\x12\x0e\n\x06values\x18\x01 \x03(\x02\"\x18\n\x08\x46ileData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"@\n\x16ImageProcessingRequest\x12\r\n\x05image\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xa8\x01\n\x17ImageProcessingResponse\x12\x15\n\rresized_image\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"G\n\x18SentimentAnalysisRequest\x12\x12\n\ninput_text\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xd3\x01\n\x19SentimentAnalysisResponse\x12\x16\n\x0esentence_count\x18\x01 \x01(\x05\x12\x10\n\x08polarity\x18\x02 \x01(\x02\x12\x14\n\x0csubjectivity\x18\x03 \x01(\x02\x12\x17\n\x0frequest_time_ms\x18\x04 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x05 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x06 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x07 \x01(\x03\"-\n\nAudioChunk\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x11\n\ttimestamp\x18\x02 \x01(\x03\"=\n\x13SpeechToTextRequest\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xae\x01\n\x14SpeechToTextResponse\x12\x1e\n\x16text_conversion_output\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"D\n\x1aImageClassificationRequest\x12\r\n\x05image\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xd0\x01\n\x1bImageClassificationResponse\x12\x17\n\x0ftop_category_id\x18\x01 \x01(\x05\x12 \n\x18top_category_probability\x18\x02 \x01(\x05\x12\x17\n\x0frequest_time_ms\x18\x03 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x04 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x05 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x06 \x01(\x03\"@\n\x16ObjectDetectionRequest\x12\r\n\x05image\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xca\x01\n\x17ObjectDetectionResponse\x12\x17\n\x0frequest_time_ms\x18\x01 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x02 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x03 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x04 \x01(\x03\x12\x37\n\x10\x64\x65tected_objects\x18\x05 \x03(\x0b\x32\x1d.protos.DetectedTrackedObject\"=\n\x13PocketSphinxRequest\x12\r\n\x05\x61udio\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\"\xa9\x01\n\x14PocketSphinxResponse\x12\x19\n\x11\x63onversion_result\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"N\n\x10\x41udioTextRequest\x12\x17\n\x0frequest_time_ms\x18\x01 \x01(\x03\x12\x12\n\ntext_input\x18\x02 \x01(\t\x12\r\n\x05\x61udio\x18\x03 \x01(\t\"\xa5\x01\n\x11\x41udioTextResponse\x12\x18\n\x10\x61lignment_result\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\"?\n\x15ObjectTrackingRequest\x12\r\n\x05image\x18\x01 \x01(\t\x12\x17\n\x0frequest_time_ms\x18\x03 \x01(\x03\"\xc8\x01\n\x16ObjectTrackingResponse\x12\x17\n\x0frequest_time_ms\x18\x01 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x02 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x03 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x04 \x01(\x03\x12\x36\n\x0ftracked_objects\x18\x05 \x03(\x0b\x32\x1d.protos.DetectedTrackedObject\")\n\x16ResourceTracingRequest\x12\x0f\n\x07timeout\x18\x01 \x01(\x05\"\xfa\x01\n\x1bResourceUtilizationResponse\x12\x1f\n\x17\x61verage_cpu_utilization\x18\x01 \x01(\x02\x12\"\n\x1a\x61verage_memory_utilization\x18\x02 \x01(\x02\x12 \n\x18\x61verage_disk_utilization\x18\x03 \x01(\x02\x12&\n\x1e\x61verage_network_received_speed\x18\x04 \x01(\x02\x12)\n!average_network_transmitted_speed\x18\x05 \x01(\x02\x12!\n\x19\x61verage_power_consumption\x18\x06 \x01(\x02\";\n\x0c\x46\x61ultRequest\x12\x15\n\rfault_command\x18\x01 \x01(\t\x12\x14\n\x0c\x66\x61ult_config\x18\x02 \x01(\t\"S\n\x15\x46\x61ultRequestWithDelay\x12\x15\n\rfault_command\x18\x01 \x01(\t\x12\x14\n\x0c\x66\x61ult_config\x18\x02 \x01(\t\x12\r\n\x05\x64\x65lay\x18\x03 \x01(\x05\"\xac\x02\n\x0cResourceLogs\x12!\n\x07\x63pu_log\x18\x01 \x01(\x0b\x32\x10.protos.FileData\x12$\n\nmemory_log\x18\x02 \x01(\x0b\x32\x10.protos.FileData\x12 \n\x06io_log\x18\x03 \x01(\x0b\x32\x10.protos.FileData\x12%\n\x0bnetwork_log\x18\x04 \x01(\x0b\x32\x10.protos.FileData\x12&\n\x1e\x66\x61ult_injection_start_times_ms\x18\x05 \x03(\x03\x12%\n\x1d\x66\x61ult_injection_stop_times_ms\x18\x06 \x03(\x03\x12!\n\x19temperature_timestamps_ms\x18\x07 \x03(\x03\x12\x18\n\x10\x63pu_temperatures\x18\x08 \x03(\x02\"$\n\rProcessStatus\x12\x13\n\x0bis_finished\x18\x01 \x01(\x08\"C\n\x07Request\x12\r\n\x05image\x18\x01 \x01(\t\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\x03\x12\x17\n\x0frequest_time_ms\x18\x03 \x01(\x03\"\xde\x01\n\x19\x44\x65tectionTrackingResponse\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12\x17\n\x0frequest_time_ms\x18\x02 \x01(\x03\x12 \n\x18request_received_time_ms\x18\x03 \x01(\x03\x12\x18\n\x10response_time_ms\x18\x04 \x01(\x03\x12!\n\x19response_received_time_ms\x18\x05 \x01(\x03\x12\x37\n\x10\x64\x65tected_objects\x18\x06 \x03(\x0b\x32\x1d.protos.DetectedTrackedObject\"t\n\x15\x44\x65tectedTrackedObject\x12\x10\n\x08track_id\x18\x01 \x01(\x05\x12\r\n\x05\x63lazz\x18\x02 \x01(\t\x12\r\n\x05x_min\x18\x03 \x01(\x05\x12\r\n\x05x_max\x18\x04 \x01(\x05\x12\r\n\x05y_min\x18\x05 \x01(\x05\x12\r\n\x05y_max\x18\x06 \x01(\x05\"\x1c\n\x08\x43PUTrace\x12\x10\n\x08\x63pu_load\x18\x01 \x01(\x02\"@\n\x0bMemoryTrace\x12\x19\n\x11\x63urrent_memory_mb\x18\x01 \x01(\x02\x12\x16\n\x0epeak_memory_mb\x18\x02 \x01(\x02\"\x0c\n\nEmptyProto2\xb5\x0c\n\x15\x41pplicationBenchmarks\x12U\n\x10image_processing\x12\x1e.protos.ImageProcessingRequest\x1a\x1f.protos.ImageProcessingResponse\"\x00\x12[\n\x12sentiment_analysis\x12 .protos.SentimentAnalysisRequest\x1a!.protos.SentimentAnalysisResponse\"\x00\x12O\n\x0espeech_to_text\x12\x1b.protos.SpeechToTextRequest\x1a\x1c.protos.SpeechToTextResponse\"\x00(\x01\x12\x41\n\x13sample_audio_stream\x12\x12.protos.AudioChunk\x1a\x12.protos.EmptyProto\"\x00(\x01\x12i\n\x1cimage_classification_alexnet\x12\".protos.ImageClassificationRequest\x1a#.protos.ImageClassificationResponse\"\x00\x12m\n image_classification_alexnet_gpu\x12\".protos.ImageClassificationRequest\x1a#.protos.ImageClassificationResponse\"\x00\x12l\n\x1fimage_classification_squeezenet\x12\".protos.ImageClassificationRequest\x1a#.protos.ImageClassificationResponse\"\x00\x12p\n#image_classification_squeezenet_gpu\x12\".protos.ImageClassificationRequest\x1a#.protos.ImageClassificationResponse\"\x00\x12]\n\x18object_detection_darknet\x12\x1e.protos.ObjectDetectionRequest\x1a\x1f.protos.ObjectDetectionResponse\"\x00\x12\x61\n\x1cobject_detection_darknet_gpu\x12\x1e.protos.ObjectDetectionRequest\x1a\x1f.protos.ObjectDetectionResponse\"\x00\x12L\n\rpocket_sphinx\x12\x1b.protos.PocketSphinxRequest\x1a\x1c.protos.PocketSphinxResponse\"\x00\x12?\n\x06\x61\x65neas\x12\x18.protos.AudioTextRequest\x1a\x19.protos.AudioTextResponse\"\x00\x12R\n\x0fobject_tracking\x12\x1d.protos.ObjectTrackingRequest\x1a\x1e.protos.ObjectTrackingResponse\"\x00\x12V\n\x13object_tracking_gpu\x12\x1d.protos.ObjectTrackingRequest\x1a\x1e.protos.ObjectTrackingResponse\"\x00\x12\x45\n\rtrack_objects\x12\x0f.protos.Request\x1a!.protos.DetectionTrackingResponse\"\x00\x12\x46\n\x0e\x64\x65tect_objects\x12\x0f.protos.Request\x1a!.protos.DetectionTrackingResponse\"\x00\x12?\n\x0cpocketsphinx\x12\x0f.protos.Request\x1a\x1c.protos.PocketSphinxResponse\"\x00\x12M\n\x11\x61lign_speech_text\x12\x18.protos.AudioTextRequest\x1a\x1c.protos.PocketSphinxResponse\"\x00\x32\xa6\x04\n\x0fMicroBenchmarks\x12`\n\x11multiply_matrices\x12#.protos.MatrixMultiplicationRequest\x1a$.protos.MatrixMultiplicationResponse\"\x00\x12S\n\x16\x66\x61st_fourier_transform\x12\x1a.protos.FastFourierRequest\x1a\x1b.protos.FastFourierResponse\"\x00\x12T\n\x13\x66loating_point_sqrt\x12\x1c.protos.FloatingPointRequest\x1a\x1d.protos.FloatingPointResponse\"\x00\x12S\n\x12\x66loating_point_sin\x12\x1c.protos.FloatingPointRequest\x1a\x1d.protos.FloatingPointResponse\"\x00\x12\x44\n\tsort_file\x12\x19.protos.FileSorterRequest\x1a\x1a.protos.FileSorterResponse\"\x00\x12\x31\n\x06\x64\x64_cmd\x12\x11.protos.DDRequest\x1a\x12.protos.DDResponse\"\x00\x12\x38\n\trun_iperf\x12\x14.protos.IperfRequest\x1a\x13.protos.IperfResult\"\x00\x32\xef\x06\n\x16\x45\x64geResourceManagement\x12\x42\n\x16start_resource_tracing\x12\x12.protos.EmptyProto\x1a\x12.protos.EmptyProto\"\x00\x12Y\n!start_resource_tracing_and_saving\x12\x1e.protos.ResourceTracingRequest\x1a\x12.protos.EmptyProto\"\x00\x12U\n\x18get_resource_utilization\x12\x12.protos.EmptyProto\x1a#.protos.ResourceUtilizationResponse\"\x00\x12I\n\x1aget_fault_injection_status\x12\x12.protos.EmptyProto\x1a\x15.protos.ProcessStatus\"\x00\x12J\n\x1bget_resource_tracing_status\x12\x12.protos.EmptyProto\x1a\x15.protos.ProcessStatus\"\x00\x12:\n\x0cinject_fault\x12\x14.protos.FaultRequest\x1a\x12.protos.EmptyProto\"\x00\x12@\n\x14stop_fault_injection\x12\x12.protos.EmptyProto\x1a\x12.protos.EmptyProto\"\x00\x12O\n\x18inject_fault_after_delay\x12\x1d.protos.FaultRequestWithDelay\x1a\x12.protos.EmptyProto\"\x00\x12?\n\x11get_resource_logs\x12\x12.protos.EmptyProto\x1a\x14.protos.ResourceLogs\"\x00\x12@\n\x14start_memory_tracing\x12\x12.protos.EmptyProto\x1a\x12.protos.EmptyProto\"\x00\x12\x37\n\rget_cpu_trace\x12\x12.protos.EmptyProto\x1a\x10.protos.CPUTrace\"\x00\x12=\n\x10get_memory_usage\x12\x12.protos.EmptyProto\x1a\x13.protos.MemoryTrace\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'benchmark_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MATRIXMULTIPLICATIONREQUEST._serialized_start=27
  _MATRIXMULTIPLICATIONREQUEST._serialized_end=149
  _MATRIXMULTIPLICATIONRESPONSE._serialized_start=152
  _MATRIXMULTIPLICATIONRESPONSE._serialized_end=334
  _FASTFOURIERREQUEST._serialized_start=336
  _FASTFOURIERREQUEST._serialized_end=418
  _FASTFOURIERRESPONSE._serialized_start=421
  _FASTFOURIERRESPONSE._serialized_end=599
  _FLOATINGPOINTREQUEST._serialized_start=601
  _FLOATINGPOINTREQUEST._serialized_end=678
  _FLOATINGPOINTRESPONSE._serialized_start=681
  _FLOATINGPOINTRESPONSE._serialized_end=855
  _FILESORTERREQUEST._serialized_start=857
  _FILESORTERREQUEST._serialized_end=933
  _FILESORTERRESPONSE._serialized_start=936
  _FILESORTERRESPONSE._serialized_end=1115
  _DDREQUEST._serialized_start=1117
  _DDREQUEST._serialized_end=1153
  _DDRESPONSE._serialized_start=1156
  _DDRESPONSE._serialized_end=1288
  _IPERFREQUEST._serialized_start=1290
  _IPERFREQUEST._serialized_end=1379
  _IPERFRESULT._serialized_start=1382
  _IPERFRESULT._serialized_end=1534
  _MATRIX._serialized_start=1536
  _MATRIX._serialized_end=1571
  _ROW._serialized_start=1573
  _ROW._serialized_end=1594
  _FILEDATA._serialized_start=1596
  _FILEDATA._serialized_end=1620
  _IMAGEPROCESSINGREQUEST._serialized_start=1622
  _IMAGEPROCESSINGREQUEST._serialized_end=1686
  _IMAGEPROCESSINGRESPONSE._serialized_start=1689
  _IMAGEPROCESSINGRESPONSE._serialized_end=1857
  _SENTIMENTANALYSISREQUEST._serialized_start=1859
  _SENTIMENTANALYSISREQUEST._serialized_end=1930
  _SENTIMENTANALYSISRESPONSE._serialized_start=1933
  _SENTIMENTANALYSISRESPONSE._serialized_end=2144
  _AUDIOCHUNK._serialized_start=2146
  _AUDIOCHUNK._serialized_end=2191
  _SPEECHTOTEXTREQUEST._serialized_start=2193
  _SPEECHTOTEXTREQUEST._serialized_end=2254
  _SPEECHTOTEXTRESPONSE._serialized_start=2257
  _SPEECHTOTEXTRESPONSE._serialized_end=2431
  _IMAGECLASSIFICATIONREQUEST._serialized_start=2433
  _IMAGECLASSIFICATIONREQUEST._serialized_end=2501
  _IMAGECLASSIFICATIONRESPONSE._serialized_start=2504
  _IMAGECLASSIFICATIONRESPONSE._serialized_end=2712
  _OBJECTDETECTIONREQUEST._serialized_start=2714
  _OBJECTDETECTIONREQUEST._serialized_end=2778
  _OBJECTDETECTIONRESPONSE._serialized_start=2781
  _OBJECTDETECTIONRESPONSE._serialized_end=2983
  _POCKETSPHINXREQUEST._serialized_start=2985
  _POCKETSPHINXREQUEST._serialized_end=3046
  _POCKETSPHINXRESPONSE._serialized_start=3049
  _POCKETSPHINXRESPONSE._serialized_end=3218
  _AUDIOTEXTREQUEST._serialized_start=3220
  _AUDIOTEXTREQUEST._serialized_end=3298
  _AUDIOTEXTRESPONSE._serialized_start=3301
  _AUDIOTEXTRESPONSE._serialized_end=3466
  _OBJECTTRACKINGREQUEST._serialized_start=3468
  _OBJECTTRACKINGREQUEST._serialized_end=3531
  _OBJECTTRACKINGRESPONSE._serialized_start=3534
  _OBJECTTRACKINGRESPONSE._serialized_end=3734
  _RESOURCETRACINGREQUEST._serialized_start=3736
  _RESOURCETRACINGREQUEST._serialized_end=3777
  _RESOURCEUTILIZATIONRESPONSE._serialized_start=3780
  _RESOURCEUTILIZATIONRESPONSE._serialized_end=4030
  _FAULTREQUEST._serialized_start=4032
  _FAULTREQUEST._serialized_end=4091
  _FAULTREQUESTWITHDELAY._serialized_start=4093
  _FAULTREQUESTWITHDELAY._serialized_end=4176
  _RESOURCELOGS._serialized_start=4179
  _RESOURCELOGS._serialized_end=4479
  _PROCESSSTATUS._serialized_start=4481
  _PROCESSSTATUS._serialized_end=4517
  _REQUEST._serialized_start=4519
  _REQUEST._serialized_end=4586
  _DETECTIONTRACKINGRESPONSE._serialized_start=4589
  _DETECTIONTRACKINGRESPONSE._serialized_end=4811
  _DETECTEDTRACKEDOBJECT._serialized_start=4813
  _DETECTEDTRACKEDOBJECT._serialized_end=4929
  _CPUTRACE._serialized_start=4931
  _CPUTRACE._serialized_end=4959
  _MEMORYTRACE._serialized_start=4961
  _MEMORYTRACE._serialized_end=5025
  _EMPTYPROTO._serialized_start=5027
  _EMPTYPROTO._serialized_end=5039
  _APPLICATIONBENCHMARKS._serialized_start=5042
  _APPLICATIONBENCHMARKS._serialized_end=6631
  _MICROBENCHMARKS._serialized_start=6634
  _MICROBENCHMARKS._serialized_end=7184
  _EDGERESOURCEMANAGEMENT._serialized_start=7187
  _EDGERESOURCEMANAGEMENT._serialized_end=8066
# @@protoc_insertion_point(module_scope)
