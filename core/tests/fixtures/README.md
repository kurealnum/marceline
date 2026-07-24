# Test fixtures

`speech_sample.wav`: a 5s slice (offset 30s, past the file's quieter
opening) of `examples/c++/aepyx.wav` from
[snakers4/silero-vad](https://github.com/snakers4/silero-vad) (MIT
licensed), 16kHz mono. Used by `vad_integration.rs` as a real speech
sample to verify Silero VAD actually detects speech, not just silence.
