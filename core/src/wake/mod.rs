//! Wake-word detection (SPEC.md §2.6, §9.6, EPIC 2.1).
//!
//! **Placeholder model, by design.** The real wake word is
//! [openWakeWord](https://github.com/dscripka/openWakeWord), but it ships
//! no "Marceline"/"Marcy" model — those are trained and exported to ONNX
//! by EPIC 13.2, which hasn't landed. This module is built so that swap
//! is a one-place change: everything here (config wiring, 16kHz mono
//! resampling, event emission/logging) is real and permanent; only
//! [`WakeDetector`]'s implementation is a stand-in.
//!
//! [`EnergyWakeDetector`] is that stand-in. It cannot discriminate one
//! spoken word from another — it fires on sustained loud audio,
//! regardless of content — so it does *not* satisfy "an unrelated word
//! does not fire" from the story's Done-when criteria. It exists to prove
//! the surrounding pipeline (resample -> detect -> event -> log) end to
//! end on real hardware. True word discrimination arrives when a real
//! ONNX-backed `WakeDetector` replaces it after EPIC 13.2.

pub mod detector;
pub mod engine;

pub use detector::{EnergyWakeDetector, WakeDetector};
pub use engine::WakeEngine;

/// The sample rate openWakeWord (and this placeholder) expects.
pub const WAKE_SAMPLE_RATE: u32 = 16_000;

/// A fired wake event: which configured word matched, its score, and when.
#[derive(Debug, Clone)]
pub struct WakeEvent {
    /// The configured word this event is attributed to. With a real
    /// per-word model this is the word that actually matched; with
    /// [`EnergyWakeDetector`] it's always the first configured word,
    /// since the placeholder can't discriminate.
    pub word: String,
    /// Detector score that crossed the fire threshold, in `[0, 1]`.
    pub score: f32,
    /// Wall-clock time the event fired.
    pub timestamp: std::time::SystemTime,
}
