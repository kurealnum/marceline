import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TtsRequest(_message.Message):
    __slots__ = ("text", "cancel")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    text: str
    cancel: _common_pb2.Cancel
    def __init__(self, text: _Optional[str] = ..., cancel: _Optional[_Union[_common_pb2.Cancel, _Mapping]] = ...) -> None: ...

class TtsResponse(_message.Message):
    __slots__ = ("audio",)
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    audio: _common_pb2.AudioChunk
    def __init__(self, audio: _Optional[_Union[_common_pb2.AudioChunk, _Mapping]] = ...) -> None: ...
