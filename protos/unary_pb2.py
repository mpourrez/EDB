# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/unary.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12protos/unary.proto\"\x1a\n\x07Message\x12\x0f\n\x07message\x18\x01 \x01(\t\"4\n\x0fMessageResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x10\n\x08received\x18\x02 \x01(\x08\x32:\n\x05Unary\x12\x31\n\x11GetServerResponse\x12\x08.Message\x1a\x10.MessageResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.unary_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MESSAGE._serialized_start=22
  _MESSAGE._serialized_end=48
  _MESSAGERESPONSE._serialized_start=50
  _MESSAGERESPONSE._serialized_end=102
  _UNARY._serialized_start=104
  _UNARY._serialized_end=162
# @@protoc_insertion_point(module_scope)
