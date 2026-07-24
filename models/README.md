# Vendored models

`silero_vad.onnx`: [Silero VAD](https://github.com/snakers4/silero-vad)
combined ONNX model (MIT licensed), used by `core::vad` (EPIC 2.2) for
endpointing. Real, off-the-shelf model — not a placeholder.

Expects: `input` (16kHz mono f32, 512 samples + 64-sample context from
the previous frame prepended, per the upstream `OnnxWrapper` reference),
`state` (`[2,1,128]` f32 recurrent state), `sr` (int64 scalar, 16000).
Outputs: `output` (`[1,1]` f32 speech probability), `stateN` (updated
state).

Wake-word models (openWakeWord "Marceline"/"Marcy") are *not* here yet —
those are trained and exported by EPIC 13.2.
