//! Loads and runs the Silero VAD ONNX model (SPEC.md §2.6, §9.3, EPIC
//! 2.2) via `ort`. Unlike the wake-word models (blocked on EPIC 13),
//! Silero VAD is a real, off-the-shelf, permissively-licensed model
//! vendored at `models/silero_vad.onnx` — this is genuine inference, not
//! a placeholder.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

/// Silero VAD's fixed input chunk size at 16kHz: exactly 512 samples
/// (32ms) per call. The model rejects any other length.
pub const FRAME_SAMPLES: usize = 512;

/// Sample rate Silero VAD expects. Callers must resample to this before
/// calling [`SileroVad::infer`].
pub const SAMPLE_RATE: i64 = 16_000;

/// Size of the model's recurrent state tensor: `[2, 1, 128]`.
const STATE_SIZE: usize = 2 * 128;

/// Trailing samples from the *previous* frame that must be prepended to
/// the current one (per Silero's own `OnnxWrapper` reference: `x =
/// cat([context, x])`, `context = x[-CONTEXT_SIZE:]`). Omitting this
/// context misaligns the model's internal convolution window against
/// what it was trained on — empirically it collapses every output to a
/// near-constant ~0.0005 regardless of input, silently, with no error.
const CONTEXT_SIZE: usize = 64;

/// Errors from loading or running the Silero VAD model.
#[derive(Debug, thiserror::Error)]
pub enum VadError {
    /// Loading the ONNX model failed (missing file, bad format, etc.).
    #[error("failed to load Silero VAD model: {0}")]
    Load(#[source] ort::Error),
    /// Running inference failed.
    #[error("VAD inference failed: {0}")]
    Inference(#[source] ort::Error),
    /// [`SileroVad::infer`] was called with a frame that isn't exactly
    /// [`FRAME_SAMPLES`] long.
    #[error("frame must be exactly {FRAME_SAMPLES} samples (16kHz), got {0}")]
    WrongFrameLen(usize),
}

/// A loaded Silero VAD session, carrying its recurrent state and
/// cross-frame context across calls.
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    /// Last [`CONTEXT_SIZE`] samples of the previous frame; prepended to
    /// the next one. Zeroed at construction/[`reset`](Self::reset), which
    /// matches the reference wrapper's behavior on the very first frame.
    context: Vec<f32>,
}

impl SileroVad {
    /// Loads the model from `model_path` (e.g. `models/silero_vad.onnx`).
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, VadError> {
        let session = Session::builder()
            .map_err(VadError::Load)?
            .commit_from_file(model_path)
            .map_err(VadError::Load)?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_SIZE],
            context: vec![0.0; CONTEXT_SIZE],
        })
    }

    /// Resets recurrent state and cross-frame context to zero, e.g. at the
    /// start of a new utterance so a prior one doesn't bias it.
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|v| *v = 0.0);
        self.context.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Runs one [`FRAME_SAMPLES`]-length, 16kHz mono frame through the
    /// model, returning the speech probability in `[0, 1]`. Updates the
    /// carried recurrent state and context for the next call.
    pub fn infer(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        if frame.len() != FRAME_SAMPLES {
            return Err(VadError::WrongFrameLen(frame.len()));
        }

        let mut windowed = Vec::with_capacity(CONTEXT_SIZE + FRAME_SAMPLES);
        windowed.extend_from_slice(&self.context);
        windowed.extend_from_slice(frame);
        self.context.copy_from_slice(&windowed[windowed.len() - CONTEXT_SIZE..]);

        let input_len = windowed.len() as i64;
        let input =
            Tensor::from_array(([1i64, input_len], windowed)).map_err(VadError::Inference)?;
        let state_tensor = Tensor::from_array(([2i64, 1, 128], self.state.clone()))
            .map_err(VadError::Inference)?;
        let sr = Tensor::from_array((Vec::<i64>::new(), vec![SAMPLE_RATE]))
            .map_err(VadError::Inference)?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state_tensor, "sr" => sr])
            .map_err(VadError::Inference)?;

        let (_, prob) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(VadError::Inference)?;
        let (_, new_state) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(VadError::Inference)?;
        self.state.copy_from_slice(new_state);

        Ok(prob[0])
    }
}
