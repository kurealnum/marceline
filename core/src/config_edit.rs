//! In-place editing of `config.toml` (EPIC 3.4; the wider config CLI is
//! EPIC 11.2).
//!
//! Separate from [`crate::config`] on purpose: that module *reads* config
//! into typed structs, which is lossy — comments, key order, and formatting
//! do not survive a deserialize/serialize round trip. `config.toml` is a
//! file a human wrote and will read again, so writing a single value back
//! must not reflow the rest of it. `toml_edit` preserves the document; a
//! `toml::to_string` of a parsed [`crate::Config`] would not.

use std::fs;
use std::path::{Path, PathBuf};

/// Errors that can occur editing a config file.
#[derive(Debug, thiserror::Error)]
pub enum ConfigEditError {
    /// The config file could not be read.
    #[error("failed to read config file {path}: {source}")]
    Read {
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
        /// Underlying parse error.
        #[source]
        source: toml_edit::TomlError,
    },
    /// The config file could not be written back.
    #[error("failed to write config file {path}: {source}")]
    Write {
        /// Path that failed to write.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The dotted key did not name a settable value.
    #[error("{key} is not a settable config key")]
    UnknownKey {
        /// Key as given by the caller.
        key: String,
    },
}

/// Reads a dotted `table.key` as a string.
///
/// Returns `None` when the key is absent or is not a string. Reads through
/// the same document model as [`set_string`] so `get`/`set` cannot disagree
/// about what a key means.
pub fn get_string(path: &Path, key: &str) -> Result<Option<String>, ConfigEditError> {
    let (table, field) = split_key(key)?;
    let doc = load(path)?;
    Ok(doc
        .get(table)
        .and_then(|item| item.as_table_like())
        .and_then(|section| section.get(field))
        .and_then(|item| item.as_str())
        .map(str::to_string))
}

/// Splits a two-part dotted key, rejecting anything else.
///
/// Only `table.key` is supported, which is all `[stt]`-style config needs;
/// nested tables arrive with the full config CLI (EPIC 11.2).
fn split_key(key: &str) -> Result<(&str, &str), ConfigEditError> {
    let unknown = || ConfigEditError::UnknownKey {
        key: key.to_string(),
    };
    let (table, field) = key.split_once('.').ok_or_else(unknown)?;
    if table.is_empty() || field.is_empty() || field.contains('.') {
        return Err(unknown());
    }
    Ok((table, field))
}

/// Reads and parses the document at `path`.
fn load(path: &Path) -> Result<toml_edit::DocumentMut, ConfigEditError> {
    let text = fs::read_to_string(path).map_err(|source| ConfigEditError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    text.parse::<toml_edit::DocumentMut>()
        .map_err(|source| ConfigEditError::Parse {
            path: path.to_path_buf(),
            source,
        })
}

/// Sets a dotted `table.key` to a string value, preserving the rest of the
/// document.
///
/// Returns the previous value, or `None` if the key was absent. Only
/// two-part keys (`stt.model`) are supported, which is all `[stt]`-style
/// config needs; nested tables arrive with the full config CLI (EPIC 11.2).
///
/// The write is atomic-ish: the new document is written to a sibling
/// temporary file and renamed over the original, so an interrupted write
/// cannot leave a truncated config behind — losing a user's whole config
/// to a half-finished `set` would be a rotten trade for one changed line.
pub fn set_string(
    path: &Path,
    key: &str,
    value: &str,
) -> Result<Option<String>, ConfigEditError> {
    let (table, field) = split_key(key)?;
    let mut doc = load(path)?;

    let Some(section) = doc.get_mut(table).and_then(|item| item.as_table_like_mut()) else {
        return Err(ConfigEditError::UnknownKey {
            key: key.to_string(),
        });
    };

    let previous = section
        .get(field)
        .and_then(|item| item.as_str())
        .map(str::to_string);

    let Some(item) = section.get_mut(field).and_then(|item| item.as_value_mut()) else {
        return Err(ConfigEditError::UnknownKey {
            key: key.to_string(),
        });
    };

    // Carry the old item's decor across, so the alignment and any trailing
    // comment on that line survive the edit.
    let mut replacement = toml_edit::Value::from(value);
    *replacement.decor_mut() = item.decor().clone();
    *item = replacement;

    write_atomically(path, &doc.to_string())?;
    Ok(previous)
}

/// Writes `contents` to `path` via a temporary file and a rename.
fn write_atomically(path: &Path, contents: &str) -> Result<(), ConfigEditError> {
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, contents).map_err(|source| ConfigEditError::Write {
        path: temp.clone(),
        source,
    })?;
    fs::rename(&temp, path).map_err(|source| ConfigEditError::Write {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_config(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "marceline-config-edit-{}-{name}.toml",
            std::process::id()
        ));
        fs::write(&path, contents).expect("write test config");
        path
    }

    const SAMPLE: &str = r#"version = 1

[stt]
backend = "whisper"             # whisper (HF default) | faster-whisper
model   = "large-v3"
device  = "cuda"
lang    = "en"                   # v1 is English-only

[tts]
backend = "kokoro"
"#;

    #[test]
    fn sets_a_value_and_returns_the_previous_one() {
        let path = temp_config("set", SAMPLE);
        let previous = set_string(&path, "stt.model", "small.en").expect("set should succeed");

        assert_eq!(previous.as_deref(), Some("large-v3"));
        let updated = fs::read_to_string(&path).unwrap();
        assert!(updated.contains(r#"model   = "small.en""#), "{updated}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn preserves_comments_and_untouched_keys() {
        // The whole reason this module exists rather than reusing the
        // serde round trip: a human's config file must survive an edit.
        let path = temp_config("preserve", SAMPLE);
        set_string(&path, "stt.model", "medium").expect("set should succeed");

        let updated = fs::read_to_string(&path).unwrap();
        assert!(
            updated.contains("# whisper (HF default) | faster-whisper"),
            "comments must survive: {updated}"
        );
        assert!(updated.contains("# v1 is English-only"), "{updated}");
        assert!(updated.contains(r#"[tts]"#));
        assert!(updated.starts_with("version = 1"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn changing_the_backend_works_too() {
        let path = temp_config("backend", SAMPLE);
        let previous =
            set_string(&path, "stt.backend", "faster-whisper").expect("set should succeed");

        assert_eq!(previous.as_deref(), Some("whisper"));
        let updated = fs::read_to_string(&path).unwrap();
        assert!(updated.contains(r#""faster-whisper""#), "{updated}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_an_unknown_table_or_key() {
        let path = temp_config("unknown", SAMPLE);

        assert!(matches!(
            set_string(&path, "nope.model", "x"),
            Err(ConfigEditError::UnknownKey { .. })
        ));
        assert!(matches!(
            set_string(&path, "stt.nonesuch", "x"),
            Err(ConfigEditError::UnknownKey { .. })
        ));
        assert!(matches!(
            set_string(&path, "sttmodel", "x"),
            Err(ConfigEditError::UnknownKey { .. })
        ));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn leaves_no_temporary_file_behind() {
        let path = temp_config("atomic", SAMPLE);
        set_string(&path, "stt.model", "tiny").expect("set should succeed");

        assert!(!path.with_extension("toml.tmp").exists());

        let _ = fs::remove_file(&path);
    }
}
