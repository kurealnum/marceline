//! Versioned `config.toml` loading, validation, and migration (SPEC.md §3.1).
//!
//! `config.toml` holds machine/runtime knobs only — never persona (that's
//! `SOUL.md`) and never secrets inline (only `*_env` pointers resolved from
//! the environment at load time).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::device::Device;

/// Current config schema version. Bump when adding a migration step.
pub const CURRENT_VERSION: u32 = 1;

/// Errors that can occur while loading, migrating, or validating a config file.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file could not be read from disk.
    #[error("failed to read config file {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The config file is not valid TOML.
    #[error("failed to parse config file {path}: {source}")]
    Parse {
        /// Path that failed to parse.
        path: PathBuf,
        /// Underlying TOML parse error.
        #[source]
        source: toml::de::Error,
    },
    /// The config's `version` field is newer than this build supports.
    #[error("config version {found} is newer than the highest supported version {max}")]
    UnsupportedVersion {
        /// Version found in the file.
        found: u32,
        /// Highest version this build knows how to load.
        max: u32,
    },
    /// The config failed semantic validation after parsing.
    #[error("config validation failed: {0}")]
    Validation(String),
    /// A `*_env` pointer referenced an environment variable that isn't set.
    #[error("environment variable {0} referenced by config is not set")]
    MissingEnv(String),
}

/// Speech-to-text backend configuration (`[stt]`).
#[derive(Debug, Clone, Deserialize)]
pub struct SttConfig {
    /// STT backend: `whisper` (HF default) or `faster-whisper`.
    pub backend: String,
    /// Model identifier/name.
    pub model: String,
    /// Compute device. v1: cuda only (device seam is story 0.7).
    pub device: Device,
    /// Recognition language. v1 is English-only.
    pub lang: String,
    /// Silence/hallucination guard thresholds (EPIC 3.6).
    ///
    /// Defaults when absent, so a config file written before these knobs
    /// existed still loads — and still gets a guard, since silently running
    /// unguarded would be the worse failure.
    #[serde(default)]
    pub guard: crate::stt::guard::GuardConfig,
}

/// LLM backend configuration (`[llm]`).
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// LLM backend, e.g. `openai-compatible`.
    pub backend: String,
    /// Base URL of the OpenAI-compatible endpoint.
    pub base_url: String,
    /// Model identifier.
    pub model: String,
    /// Name of the environment variable holding the API key (never the key itself).
    pub api_key_env: String,
    /// Max tokens generated per turn (cost guardrail).
    pub max_tokens_per_turn: u32,
    /// Max requests allowed per session (cost guardrail).
    pub max_requests_per_session: u32,
    /// Max tool-call iterations allowed per turn before forcing a final answer.
    pub max_tool_iterations_per_turn: u32,
}

impl LlmConfig {
    /// Resolves the actual API key from the environment variable named by
    /// `api_key_env`. The key is never stored in the config struct or file.
    pub fn resolve_api_key(&self) -> Result<String, ConfigError> {
        env::var(&self.api_key_env).map_err(|_| ConfigError::MissingEnv(self.api_key_env.clone()))
    }
}

/// Text-to-speech backend configuration (`[tts]`).
#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    /// TTS backend: `kokoro` or `piper`.
    pub backend: String,
    /// Backend-specific voice id.
    pub voice: String,
    /// Compute device.
    pub device: Device,
}

/// Wake word configuration (`[wake]`).
#[derive(Debug, Clone, Deserialize)]
pub struct WakeConfig {
    /// Wake words / barge-in intent words.
    pub words: Vec<String>,
    /// Detection sensitivity in `[0, 1]`.
    pub sensitivity: f64,
}

/// Voice activity detection / endpointing configuration (`[vad]`).
#[derive(Debug, Clone, Deserialize)]
pub struct VadConfig {
    /// Silence duration (ms) that ends an utterance.
    pub silence_ms: u32,
    /// Minimum utterance duration (ms).
    pub min_utterance_ms: u32,
    /// Maximum utterance duration (ms).
    pub max_utterance_ms: u32,
}

/// Memory/history store configuration (`[memory]`).
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    /// Path to the history database. `~` is expanded to the user's home directory.
    pub db_path: String,
    /// Whether long-term memory is enabled.
    pub longterm: bool,
    /// Embedding model identifier.
    pub embed_model: String,
    /// Compute device used for embedding.
    pub embed_device: Device,
}

impl MemoryConfig {
    /// Returns `db_path` with a leading `~` expanded to the user's home directory.
    pub fn expanded_db_path(&self) -> PathBuf {
        expand_tilde(&self.db_path)
    }
}

/// Egress auditing configuration (`[egress]`).
#[derive(Debug, Clone, Deserialize)]
pub struct EgressConfig {
    /// Whether to audit-log everything leaving the machine.
    pub log: bool,
}

/// Audio input/output device selection (`[audio]`, EPIC 1.3).
///
/// Absent from older config files entirely (the whole section defaults
/// via `#[serde(default)]` on [`Config::audio`]) and both fields default
/// to `None`, meaning "use the system default device."
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AudioConfig {
    /// Input device name to capture from. `None`/absent/empty resolves to
    /// the host's default input device; an unrecognized name falls back
    /// to the default with a warning (resolution happens in
    /// `core::audio`, not here — this struct only carries the request).
    #[serde(default)]
    pub input_device: Option<String>,
    /// Output device name to play to. Same fallback behavior as
    /// `input_device`.
    #[serde(default)]
    pub output_device: Option<String>,
}

/// Top-level, versioned machine/runtime configuration (`config.toml`).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Config schema version; drives migration on upgrade.
    pub version: u32,
    /// Speech-to-text settings.
    pub stt: SttConfig,
    /// LLM settings.
    pub llm: LlmConfig,
    /// Text-to-speech settings.
    pub tts: TtsConfig,
    /// Wake word settings.
    pub wake: WakeConfig,
    /// Voice activity detection settings.
    pub vad: VadConfig,
    /// Memory/history store settings.
    pub memory: MemoryConfig,
    /// Egress auditing settings.
    pub egress: EgressConfig,
    /// Audio input/output device selection. Defaults when the whole
    /// `[audio]` section is absent, so older config files keep loading.
    #[serde(default)]
    pub audio: AudioConfig,
}

impl Config {
    /// Loads, migrates (if needed), validates, and returns the config at `path`.
    ///
    /// If the file's `version` is older than [`CURRENT_VERSION`], it is
    /// migrated in place: unknown/old keys are preserved, missing
    /// sections/keys are filled with defaults, a warning is printed, and the
    /// migrated file is rewritten to disk before being parsed.
    pub fn load(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        let mut value: toml::Value = text.parse().map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;

        let found_version = value
            .get("version")
            .and_then(toml::Value::as_integer)
            .unwrap_or(0) as u32;

        if found_version > CURRENT_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: found_version,
                max: CURRENT_VERSION,
            });
        }

        if found_version < CURRENT_VERSION {
            eprintln!(
                "warning: config.toml is version {found_version}, migrating to {CURRENT_VERSION}"
            );
            migrate(&mut value, found_version);
            let rewritten = toml::to_string_pretty(&value).map_err(|e| {
                ConfigError::Validation(format!("failed to serialize migrated config: {e}"))
            })?;
            fs::write(path, rewritten).map_err(|source| ConfigError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        }

        let config: Config = value.try_into().map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;

        config.validate()?;
        Ok(config)
    }

    /// Checks semantic constraints that plain deserialization can't express.
    fn validate(&self) -> Result<(), ConfigError> {
        if self.wake.words.is_empty() {
            return Err(ConfigError::Validation(
                "wake.words must not be empty".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.wake.sensitivity) {
            return Err(ConfigError::Validation(
                "wake.sensitivity must be in [0, 1]".into(),
            ));
        }
        if self.vad.min_utterance_ms >= self.vad.max_utterance_ms {
            return Err(ConfigError::Validation(
                "vad.min_utterance_ms must be less than vad.max_utterance_ms".into(),
            ));
        }
        if self.llm.max_tokens_per_turn == 0 {
            return Err(ConfigError::Validation(
                "llm.max_tokens_per_turn must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

/// Applies migrations in sequence from `from_version` up to [`CURRENT_VERSION`],
/// mutating `value` in place. Never removes existing keys — only adds
/// defaults for sections/keys introduced by a later version.
fn migrate(value: &mut toml::Value, from_version: u32) {
    if from_version < 1 {
        migrate_0_to_1(value);
    }
    if let Some(table) = value.as_table_mut() {
        table.insert("version".into(), toml::Value::Integer(CURRENT_VERSION as i64));
    }
}

/// Migration from the unversioned/pre-v1 schema to v1: fills in any missing
/// top-level sections with their v1 defaults. Existing keys are untouched.
fn migrate_0_to_1(value: &mut toml::Value) {
    let table = match value.as_table_mut() {
        Some(t) => t,
        None => return,
    };

    let defaults: toml::Table = toml::toml! {
        [stt]
        backend = "whisper"
        model = "large-v3"
        device = "cuda"
        lang = "en"

        [llm]
        backend = "openai-compatible"
        base_url = "http://localhost:1234/v1"
        model = "local-model"
        api_key_env = "MARCELINE_LLM_KEY"
        max_tokens_per_turn = 2048
        max_requests_per_session = 200
        max_tool_iterations_per_turn = 8

        [tts]
        backend = "kokoro"
        voice = "af_sky"
        device = "cuda"

        [wake]
        words = ["marceline", "marcy"]
        sensitivity = 0.6

        [vad]
        silence_ms = 700
        min_utterance_ms = 300
        max_utterance_ms = 15000

        [memory]
        db_path = "~/.marceline/history.db"
        longterm = true
        embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        embed_device = "cpu"

        [egress]
        log = true
    };

    for (key, default_value) in defaults {
        table.entry(key).or_insert(default_value);
    }
}

/// Expands a leading `~` (or `~/...`) to the current user's home directory.
/// Paths without a leading `~` are returned unchanged.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if path == "~" {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Returns a unique scratch file path in the OS temp directory for a test run.
    fn scratch_path(name: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("marceline-config-test-{}-{n}-{name}", std::process::id()))
    }

    const VALID_V1: &str = r#"
version = 1

[stt]
backend = "whisper"
model = "large-v3"
device = "cuda"
lang = "en"

[llm]
backend = "openai-compatible"
base_url = "http://localhost:1234/v1"
model = "local-model"
api_key_env = "MARCELINE_LLM_KEY"
max_tokens_per_turn = 2048
max_requests_per_session = 200
max_tool_iterations_per_turn = 8

[tts]
backend = "kokoro"
voice = "af_sky"
device = "cuda"

[wake]
words = ["marceline", "marcy"]
sensitivity = 0.6

[vad]
silence_ms = 700
min_utterance_ms = 300
max_utterance_ms = 15000

[memory]
db_path = "~/.marceline/history.db"
longterm = true
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
embed_device = "cpu"

[egress]
log = true
"#;

    #[test]
    fn loads_valid_v1_config() {
        let path = scratch_path("valid.toml");
        fs::write(&path, VALID_V1).unwrap();
        let config = Config::load(&path).expect("valid config should load");
        assert_eq!(config.version, 1);
        assert_eq!(config.stt.backend, "whisper");
        assert_eq!(
            config.memory.expanded_db_path().to_string_lossy(),
            format!("{}/.marceline/history.db", env::var("HOME").unwrap())
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn stt_guard_defaults_when_the_section_is_absent() {
        // A config written before the guard knobs existed must still load —
        // and must still get a guard, since running unguarded silently is
        // the worse failure (EPIC 3.6).
        let path = scratch_path("guard-default.toml");
        fs::write(&path, VALID_V1).unwrap();
        let config = Config::load(&path).expect("valid config should load");

        assert_eq!(config.stt.guard, crate::stt::GuardConfig::default());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn stt_guard_thresholds_can_be_overridden_individually() {
        let path = scratch_path("guard-override.toml");
        fs::write(
            &path,
            VALID_V1.replace(
                "[llm]",
                "[stt.guard]\nmax_no_speech_prob = 0.4\n\n[llm]",
            ),
        )
        .unwrap();
        let config = Config::load(&path).expect("valid config should load");

        assert_eq!(config.stt.guard.max_no_speech_prob, 0.4);
        // Untouched knobs keep their defaults rather than zeroing out.
        assert_eq!(
            config.stt.guard.min_speech_ms,
            crate::stt::guard::DEFAULT_MIN_SPEECH_MS
        );
        assert_eq!(
            config.stt.guard.min_avg_logprob,
            crate::stt::guard::DEFAULT_MIN_AVG_LOGPROB
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_invalid_config_with_clear_error() {
        let path = scratch_path("invalid.toml");
        fs::write(&path, "version = 1\n[stt]\nbackend = \"whisper\"\n").unwrap();
        let err = Config::load(&path).expect_err("missing required fields should fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_unsupported_device_with_clear_error() {
        let path = scratch_path("bad-device.toml");
        let bad = VALID_V1.replacen("device = \"cuda\"", "device = \"metal\"", 1);
        fs::write(&path, bad).unwrap();
        let err = Config::load(&path).expect_err("unsupported device should fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_out_of_range_sensitivity() {
        let path = scratch_path("bad-sensitivity.toml");
        let bad = VALID_V1.replace("sensitivity = 0.6", "sensitivity = 1.6");
        fs::write(&path, bad).unwrap();
        let err = Config::load(&path).expect_err("out-of-range sensitivity should fail");
        assert!(matches!(err, ConfigError::Validation(_)));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn migrates_pre_v1_file_and_preserves_unknown_keys() {
        let path = scratch_path("legacy.toml");
        let legacy = r#"
custom_user_note = "do not delete me"

[stt]
backend = "faster-whisper"
model = "large-v3"
device = "cuda"
lang = "en"
"#;
        fs::write(&path, legacy).unwrap();

        let config = Config::load(&path).expect("legacy config should migrate and load");
        assert_eq!(config.version, CURRENT_VERSION);
        // Migration must not clobber values already present in the file.
        assert_eq!(config.stt.backend, "faster-whisper");

        let rewritten = fs::read_to_string(&path).unwrap();
        assert!(rewritten.contains("custom_user_note"));
        assert!(rewritten.contains("do not delete me"));
        assert!(rewritten.contains("version = 1"));
        fs::remove_file(&path).ok();
    }
}
