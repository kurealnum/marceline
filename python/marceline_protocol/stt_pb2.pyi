import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SttInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SttInfo(_message.Message):
    __slots__ = ("name", "langs", "input_sample_rate", "partials")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGS_FIELD_NUMBER: _ClassVar[int]
    INPUT_SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    PARTIALS_FIELD_NUMBER: _ClassVar[int]
    name: str
    langs: _containers.RepeatedScalarFieldContainer[str]
    input_sample_rate: int
    partials: bool
    def __init__(self, name: _Optional[str] = ..., langs: _Optional[_Iterable[str]] = ..., input_sample_rate: _Optional[int] = ..., partials: bool = ...) -> None: ...

class SttRequest(_message.Message):
    __slots__ = ("audio", "cancel")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    audio: _common_pb2.AudioChunk
    cancel: _common_pb2.Cancel
    def __init__(self, audio: _Optional[_Union[_common_pb2.AudioChunk, _Mapping]] = ..., cancel: _Optional[_Union[_common_pb2.Cancel, _Mapping]] = ...) -> None: ...

class SttResponse(_message.Message):
    __slots__ = ("partial", "final")
    PARTIAL_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    partial: str
    final: FinalTranscript
    def __init__(self, partial: _Optional[str] = ..., final: _Optional[_Union[FinalTranscript, _Mapping]] = ...) -> None: ...

class FinalTranscript(_message.Message):
    __slots__ = ("text", "confidence", "no_speech_prob", "avg_logprob")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    NO_SPEECH_PROB_FIELD_NUMBER: _ClassVar[int]
    AVG_LOGPROB_FIELD_NUMBER: _ClassVar[int]
    text: str
    confidence: float
    no_speech_prob: float
    avg_logprob: float
    def __init__(self, text: _Optional[str] = ..., confidence: _Optional[float] = ..., no_speech_prob: _Optional[float] = ..., avg_logprob: _Optional[float] = ...) -> None: ...
