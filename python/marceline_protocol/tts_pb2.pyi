import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TtsInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TtsInfo(_message.Message):
    __slots__ = ("name", "voices", "output_sample_rate")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VOICES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    voices: _containers.RepeatedScalarFieldContainer[str]
    output_sample_rate: int
    def __init__(self, name: _Optional[str] = ..., voices: _Optional[_Iterable[str]] = ..., output_sample_rate: _Optional[int] = ...) -> None: ...

class TtsRequest(_message.Message):
    __slots__ = ("text", "cancel", "voice")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    text: str
    cancel: _common_pb2.Cancel
    voice: str
    def __init__(self, text: _Optional[str] = ..., cancel: _Optional[_Union[_common_pb2.Cancel, _Mapping]] = ..., voice: _Optional[str] = ...) -> None: ...

class TtsResponse(_message.Message):
    __slots__ = ("audio",)
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    audio: _common_pb2.AudioChunk
    def __init__(self, audio: _Optional[_Union[_common_pb2.AudioChunk, _Mapping]] = ...) -> None: ...
