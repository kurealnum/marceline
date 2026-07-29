//! Integration test for the wake+VAD gate state machine (EPIC 2.3):
//! drives the real Silero VAD (behind [`VadEndpointer`]) and the
//! placeholder [`EnergyWakeDetector`] through a full wake -> listen ->
//! speech -> silence -> emit cycle, and the no-speech-timeout bailout.

use marceline_core::config::{VadConfig, WakeConfig};
use marceline_core::{
    AudioChunk, EnergyWakeDetector, Gate, GateOutput, GateState, SileroVad, VadEndpointer,
    WakeEngine, DEFAULT_SPEECH_THRESHOLD,
};

const SAMPLE_RATE: u32 = 16_000;

fn model_path() -> String {
    format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"))
}

fn default_vad_config() -> VadConfig {
    VadConfig {
        silence_ms: 700,
        min_utterance_ms: 300,
        max_utterance_ms: 15_000,
        barge_in_confirm_ms: 300,
        no_speech_timeout_ms: 3_000,
    }
}

fn build_gate() -> Gate {
    build_gate_with(&default_vad_config())
}

fn build_gate_with(vad_config: &VadConfig) -> Gate {
    let wake_config = WakeConfig {
        words: vec!["marceline".to_string()],
        sensitivity: 0.6,
    };
    let detector = EnergyWakeDetector::new(wake_config.sensitivity, SAMPLE_RATE, 320);
    let wake = WakeEngine::new(&wake_config, Box::new(detector));
    let vad = SileroVad::load(model_path()).expect("failed to load Silero VAD model");
    let endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);
    Gate::new(wake, endpointer, vad_config)
}

fn chunk(pcm: Vec<f32>) -> AudioChunk {
    AudioChunk {
        seq: 0,
        pcm,
        sample_rate: SAMPLE_RATE,
        channels: 1,
    }
}

fn silence_chunk(len: usize) -> AudioChunk {
    chunk(vec![0.0; len])
}

fn loud_tone_chunk(len: usize) -> AudioChunk {
    chunk((0..len).map(|i| 0.9 * (i as f32 * 0.3).sin()).collect())
}

fn load_speech_sample() -> Vec<f32> {
    let fixture = format!(
        "{}/tests/fixtures/speech_sample.wav",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut reader = hound::WavReader::open(&fixture).expect("failed to open speech fixture");
    reader
        .samples::<i16>()
        .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
        .collect()
}

#[test]
fn wake_then_speech_then_silence_emits_one_segment_seeded_with_preroll() {
    let mut gate = build_gate();
    let empty_preroll = silence_chunk(0);

    // IDLE: quiet chunks must not fire wake.
    for _ in 0..5 {
        let out = gate.process_chunk(&silence_chunk(320), &empty_preroll);
        assert!(matches!(out, GateOutput::None));
        assert_eq!(gate.state(), GateState::Idle);
    }

    // A short, distinct pre-roll snippet — must appear at the head of the
    // eventually-emitted segment.
    let preroll = chunk(vec![0.42; 1600]); // 100ms distinctive marker

    // Sustained loud audio fires the (placeholder) wake detector.
    let mut fired = false;
    for _ in 0..20 {
        match gate.process_chunk(&loud_tone_chunk(320), &preroll) {
            GateOutput::Wake => {
                fired = true;
                break;
            }
            GateOutput::None => {}
            other => panic!("unexpected gate output while idle: {other:?}"),
        }
    }
    assert!(fired, "sustained loud audio should fire wake");
    assert_eq!(gate.state(), GateState::Listening);

    // Feed a continuous 1.5s slice of real speech with no internal pause
    // anywhere near the 700ms silence-end threshold (verified offline) —
    // this is the "one phrase" the Done-when criterion describes. Must
    // not time out or emit mid-phrase.
    let speech = load_speech_sample();
    let continuous_phrase = &speech[..24_000.min(speech.len())];
    let mut segments_during_speech = 0;
    for frame in continuous_phrase.chunks(1600) {
        match gate.process_chunk(&chunk(frame.to_vec()), &preroll) {
            GateOutput::None => {}
            GateOutput::Segment(_) => segments_during_speech += 1,
            GateOutput::NoSpeechTimeout => panic!("should not time out mid-speech"),
            GateOutput::TooShort => panic!("continuous phrase should not be discarded as too short"),
            GateOutput::Wake => panic!("should not re-fire wake while listening"),
            GateOutput::WakeDetected => panic!("WakeDetected only fires in THINKING/SPEAKING"),
        }
    }
    assert_eq!(segments_during_speech, 0, "must not emit before trailing silence");
    assert_eq!(gate.state(), GateState::Listening);

    // Trailing silence should eventually end the utterance.
    let mut emitted = None;
    for _ in 0..100 {
        match gate.process_chunk(&silence_chunk(1600), &preroll) {
            GateOutput::Segment(segment) => {
                emitted = Some(segment);
                break;
            }
            GateOutput::None => {}
            other => panic!("unexpected gate output during trailing silence: {other:?}"),
        }
    }

    let segment = emitted.expect("expected a segment to be emitted after trailing silence");
    assert_eq!(gate.state(), GateState::Idle, "gate returns to IDLE after emitting");

    // Pre-roll marker must be at the head of the emitted segment.
    assert!(segment.pcm.len() >= preroll.pcm.len());
    assert_eq!(&segment.pcm[..preroll.pcm.len()], &preroll.pcm[..]);
    assert_eq!(segment.sample_rate, SAMPLE_RATE);
    assert_eq!(segment.channels, 1);
}

#[test]
fn wake_with_no_following_speech_times_out_back_to_idle() {
    let mut gate = build_gate();
    let empty_preroll = silence_chunk(0);

    let mut fired = false;
    for _ in 0..20 {
        if matches!(
            gate.process_chunk(&loud_tone_chunk(320), &empty_preroll),
            GateOutput::Wake
        ) {
            fired = true;
            break;
        }
    }
    assert!(fired);
    assert_eq!(gate.state(), GateState::Listening);

    // Feed silence past the no-speech timeout; the gate must bail to IDLE
    // rather than waiting forever or crashing.
    let mut timed_out = false;
    for _ in 0..200 {
        match gate.process_chunk(&silence_chunk(1600), &empty_preroll) {
            GateOutput::NoSpeechTimeout => {
                timed_out = true;
                break;
            }
            GateOutput::Segment(_) => panic!("silence alone must never emit a segment"),
            GateOutput::TooShort => panic!("no speech was ever heard; TooShort shouldn't fire"),
            GateOutput::None => {}
            GateOutput::Wake => panic!("should not re-fire wake while listening"),
            GateOutput::WakeDetected => panic!("WakeDetected only fires in THINKING/SPEAKING"),
        }
    }
    assert!(timed_out, "expected a no-speech timeout");
    assert_eq!(gate.state(), GateState::Idle);
}

#[test]
fn min_utterance_ms_discards_a_short_speech_blip() {
    // A tight min_utterance_ms (500ms) and a brief loud "speech" blip
    // (well under it) followed by silence should discard, not emit.
    let vad_config = VadConfig {
        silence_ms: 200,
        min_utterance_ms: 500,
        max_utterance_ms: 15_000,
        barge_in_confirm_ms: 300,
        no_speech_timeout_ms: 3_000,
    };
    let mut gate = build_gate_with(&vad_config);
    let empty_preroll = silence_chunk(0);

    let mut fired = false;
    for _ in 0..20 {
        if matches!(
            gate.process_chunk(&loud_tone_chunk(320), &empty_preroll),
            GateOutput::Wake
        ) {
            fired = true;
            break;
        }
    }
    assert!(fired);

    // ~100ms of real speech — under the 500ms min_utterance_ms floor.
    let speech = load_speech_sample();
    let brief_blip = &speech[..1_600.min(speech.len())];
    for frame in brief_blip.chunks(1600) {
        let out = gate.process_chunk(&chunk(frame.to_vec()), &empty_preroll);
        assert!(
            !matches!(out, GateOutput::Segment(_)),
            "must not emit mid-blip"
        );
    }

    let mut discarded = false;
    for _ in 0..50 {
        match gate.process_chunk(&silence_chunk(1600), &empty_preroll) {
            GateOutput::TooShort => {
                discarded = true;
                break;
            }
            GateOutput::Segment(_) => panic!("a sub-min_utterance_ms blip must not emit"),
            GateOutput::None => {}
            other => panic!("unexpected output: {other:?}"),
        }
    }
    assert!(discarded, "expected the brief blip to be discarded as too short");
    assert_eq!(gate.state(), GateState::Idle);
}

#[test]
fn max_utterance_ms_force_emits_an_overlong_segment() {
    // A tiny max_utterance_ms so continuous real speech (no trailing
    // silence at all) still gets force-emitted rather than collected
    // forever.
    let vad_config = VadConfig {
        silence_ms: 700,
        min_utterance_ms: 0,
        max_utterance_ms: 400,
        barge_in_confirm_ms: 300,
        no_speech_timeout_ms: 3_000,
    };
    let mut gate = build_gate_with(&vad_config);
    let empty_preroll = silence_chunk(0);

    let mut fired = false;
    for _ in 0..20 {
        if matches!(
            gate.process_chunk(&loud_tone_chunk(320), &empty_preroll),
            GateOutput::Wake
        ) {
            fired = true;
            break;
        }
    }
    assert!(fired);

    let speech = load_speech_sample();
    let mut emitted = false;
    for frame in speech.chunks(1600) {
        match gate.process_chunk(&chunk(frame.to_vec()), &empty_preroll) {
            GateOutput::Segment(_) => {
                emitted = true;
                break;
            }
            GateOutput::TooShort => panic!("should force-emit, not discard"),
            _ => {}
        }
    }
    assert!(emitted, "expected max_utterance_ms to force an emission");
    assert_eq!(gate.state(), GateState::Idle);
}

#[test]
fn wake_detector_stays_armed_through_thinking_and_speaking() {
    // EPIC 7.1: the gate must not go deaf the moment a turn leaves
    // IDLE/LISTENING. Verify the wake detector keeps firing in THINKING
    // and SPEAKING, without collecting an utterance or touching state.
    let mut gate = build_gate();
    let empty_preroll = silence_chunk(0);

    gate.enter_thinking();
    assert_eq!(gate.state(), GateState::Thinking);
    let mut fired = false;
    for _ in 0..20 {
        match gate.process_chunk(&loud_tone_chunk(320), &empty_preroll) {
            GateOutput::WakeDetected => {
                fired = true;
                break;
            }
            GateOutput::None => {}
            other => panic!("unexpected output while THINKING: {other:?}"),
        }
    }
    assert!(fired, "wake detector must still fire while THINKING");
    assert_eq!(
        gate.state(),
        GateState::Thinking,
        "a bare detection must not itself change state"
    );

    // Clear the wake detector's cooldown (EPIC 2.1) before expecting it
    // to fire again.
    for _ in 0..100 {
        gate.process_chunk(&silence_chunk(320), &empty_preroll);
    }

    gate.enter_speaking();
    assert_eq!(gate.state(), GateState::Speaking);
    let mut fired = false;
    for _ in 0..20 {
        match gate.process_chunk(&loud_tone_chunk(320), &empty_preroll) {
            GateOutput::WakeDetected => {
                fired = true;
                break;
            }
            GateOutput::None => {}
            other => panic!("unexpected output while SPEAKING: {other:?}"),
        }
    }
    assert!(fired, "wake detector must still fire while SPEAKING");
    assert_eq!(gate.state(), GateState::Speaking);

    gate.enter_idle();
    assert_eq!(gate.state(), GateState::Idle);
}
