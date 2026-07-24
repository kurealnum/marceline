from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AudioChunk(_message.Message):
    __slots__ = ("seq", "pcm", "sample_rate", "channels")
    SEQ_FIELD_NUMBER: _ClassVar[int]
    PCM_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    seq: int
    pcm: _containers.RepeatedScalarFieldContainer[float]
    sample_rate: int
    channels: int
    def __init__(self, seq: _Optional[int] = ..., pcm: _Optional[_Iterable[float]] = ..., sample_rate: _Optional[int] = ..., channels: _Optional[int] = ...) -> None: ...

class Cancel(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
