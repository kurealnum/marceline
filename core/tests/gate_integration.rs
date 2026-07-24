//! Integration test for the wake+VAD gate state machine (EPIC 2.3):
//! drives the real Silero VAD (behind [`VadEndpointer`]) and the
//! placeholder [`EnergyWakeDetector`] through a full wake -> listen ->
//! speech -> silence -> emit cycle, and the no-speech-timeout bailout.

use marceline_core::config::WakeConfig;
use marceline_core::{
    AudioChunk, EnergyWakeDetector, Gate, GateOutput, GateState, SileroVad, VadEndpointer,
    WakeEngine, DEFAULT_SPEECH_THRESHOLD,
};

const SAMPLE_RATE: u32 = 16_000;

fn model_path() -> String {
    format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"))
}

fn build_gate() -> Gate {
    let wake_config = WakeConfig {
        words: vec!["marceline".to_string()],
        sensitivity: 0.6,
    };
    let detector = EnergyWakeDetector::new(wake_config.sensitivity, SAMPLE_RATE, 320);
    let wake = WakeEngine::new(&wake_config, Box::new(detector));
    let vad = SileroVad::load(model_path()).expect("failed to load Silero VAD model");
    let endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);
    Gate::new(wake, endpointer)
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
            GateOutput::Wake => panic!("should not re-fire wake while listening"),
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
            GateOutput::None => {}
            GateOutput::Wake => panic!("should not re-fire wake while listening"),
        }
    }
    assert!(timed_out, "expected a no-speech timeout");
    assert_eq!(gate.state(), GateState::Idle);
}
