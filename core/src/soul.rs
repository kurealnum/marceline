//! Parses user-authored `SOUL.md` into a structured [`Persona`] and renders
//! it back to markdown for [`crate::llm::prompt::compile_system_prompt`]
//! (SPEC.md §3.2, EPIC 9.1).
//!
//! `SOUL.md` is free-form and un-versioned (unlike `config.toml`, §0.2): no
//! schema migration, no required sections, no fixed order. This module only
//! recognizes the six suggested top-level headings; anything else is text
//! the user chose to write and is preserved verbatim inside whichever
//! section it physically falls under (or dropped if it precedes the first
//! recognized heading, mirroring how an unlabeled preamble in a persona file
//! has nowhere principled to go).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::tools::SafetyClass;

/// Errors that can occur while loading a `SOUL.md` file from disk.
///
/// Parsing the text itself never fails — `SOUL.md` is free-form prose, so
/// any input, including one with missing or reordered sections, yields a
/// valid (possibly mostly-empty) [`Persona`]. Only the read from disk can
/// fail.
#[derive(Debug, thiserror::Error)]
pub enum SoulError {
    /// The SOUL.md file could not be read from disk.
    #[error("failed to read SOUL.md file {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// One of the six sections §3.2 suggests for `SOUL.md`. Order here is the
/// canonical render order used by [`Persona::render`], independent of
/// whatever order the user wrote them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Identity,
    Voice,
    ValuesRules,
    Knowledge,
    ToolsPolicy,
    Examples,
}

impl Section {
    /// Matches a markdown `#` heading's text against a recognized section,
    /// case-insensitively and tolerant of the punctuation variants §3.2
    /// itself uses (e.g. "Values / rules").
    fn from_heading(heading: &str) -> Option<Section> {
        let normalized: String = heading
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        match normalized.as_str() {
            "identity" => Some(Section::Identity),
            "voice" => Some(Section::Voice),
            "values rules" | "values" | "rules" => Some(Section::ValuesRules),
            "knowledge about me the user" | "knowledge about me" | "knowledge" => {
                Some(Section::Knowledge)
            }
            "tools policy" | "tool policy" => Some(Section::ToolsPolicy),
            "examples" => Some(Section::Examples),
            _ => None,
        }
    }

    /// The canonical heading text this section renders back to.
    fn heading(self) -> &'static str {
        match self {
            Section::Identity => "Identity",
            Section::Voice => "Voice",
            Section::ValuesRules => "Values / rules",
            Section::Knowledge => "Knowledge about me (the user)",
            Section::ToolsPolicy => "Tools policy",
            Section::Examples => "Examples",
        }
    }
}

/// A `SOUL.md` file parsed into its six suggested sections (SPEC.md §3.2).
///
/// Every field is optional: a section absent from the source file is
/// `None`, and downstream consumers (the system-prompt compiler, 9.3's tool
/// gating, 9.4's voice routing) must treat that as "the user didn't specify
/// this" rather than an error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Persona {
    /// Name, personality, speaking style, verbosity, humor.
    pub identity: Option<String>,
    /// Preferred TTS voice, pacing, terse-vs-expansive (routed to TTS by 9.4).
    pub voice: Option<String>,
    /// Hard rules, tone boundaries, safety preferences.
    pub values_rules: Option<String>,
    /// Standing facts about the user (name, timezone, projects, preferences).
    pub knowledge: Option<String>,
    /// Which tools are allowed/confirm/off (routed to 6.5 gating by 9.3).
    pub tools_policy: Option<String>,
    /// Few-shot exchanges demonstrating desired behavior.
    pub examples: Option<String>,
}

impl Persona {
    /// Loads and parses `SOUL.md` at `path`.
    ///
    /// A missing file is the caller's concern, not this function's — pass
    /// through `Config::load`-style error handling, or fall back to
    /// [`Persona::default`] at the call site the same way the pre-9.1 code
    /// treated a missing file as an empty persona.
    pub fn load(path: impl AsRef<Path>) -> Result<Persona, SoulError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| SoulError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Persona::parse(&text))
    }

    /// Parses `SOUL.md`'s raw markdown text into a [`Persona`].
    ///
    /// Sections are split on top-level (`# `) headings; anything under a
    /// heading `from_heading` doesn't recognize is ignored, and text before
    /// the first recognized heading is ignored. Missing or reordered
    /// sections never produce an error.
    pub fn parse(text: &str) -> Persona {
        let mut persona = Persona::default();
        let mut current: Option<Section> = None;
        let mut body = String::new();

        macro_rules! flush {
            () => {
                if let Some(section) = current {
                    let trimmed = body.trim();
                    if !trimmed.is_empty() {
                        let slot = match section {
                            Section::Identity => &mut persona.identity,
                            Section::Voice => &mut persona.voice,
                            Section::ValuesRules => &mut persona.values_rules,
                            Section::Knowledge => &mut persona.knowledge,
                            Section::ToolsPolicy => &mut persona.tools_policy,
                            Section::Examples => &mut persona.examples,
                        };
                        *slot = Some(trimmed.to_string());
                    }
                }
                body.clear();
            };
        }

        for line in text.lines() {
            if let Some(heading) = line.strip_prefix("# ") {
                flush!();
                current = Section::from_heading(heading);
                continue;
            }
            if current.is_some() {
                body.push_str(line);
                body.push('\n');
            }
        }
        flush!();

        persona
    }

    /// Parses this persona's Tools policy section into a [`ToolPolicy`]
    /// (§3.2, EPIC 9.3). An absent section yields an empty policy, which
    /// still gates every tool via [`ToolPolicy::decision`]'s per-class
    /// default.
    pub fn tool_policy(&self) -> ToolPolicy {
        ToolPolicy::parse(self.tools_policy.as_deref().unwrap_or(""))
    }

    /// Renders this persona back to canonical markdown, in the fixed order
    /// §3.2 suggests, for the system-prompt compiler to consume in place of
    /// `SOUL.md`'s raw text.
    ///
    /// Sections the user left out are simply omitted — the compiler must
    /// work from persona text alone, with no schema-implied placeholders
    /// leaking into the prompt.
    pub fn render(&self) -> String {
        let sections: [(Section, &Option<String>); 6] = [
            (Section::Identity, &self.identity),
            (Section::Voice, &self.voice),
            (Section::ValuesRules, &self.values_rules),
            (Section::Knowledge, &self.knowledge),
            (Section::ToolsPolicy, &self.tools_policy),
            (Section::Examples, &self.examples),
        ];

        let mut out = String::new();
        for (section, content) in sections {
            if let Some(content) = content {
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str("# ");
                out.push_str(section.heading());
                out.push_str("\n\n");
                out.push_str(content);
            }
        }
        out
    }
}

/// A per-tool decision from SOUL.md's Tools policy section (§3.2), consumed
/// by the THINKING loop's gating (EPIC 6.5/9.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolDecision {
    /// Runs without a spoken confirmation.
    Auto,
    /// Requires a spoken confirmation before running (EPIC 6.5).
    Confirm,
    /// Never offered to the model and never dispatched.
    Off,
}

/// Per-tool policy parsed from SOUL.md's `# Tools policy` section (§3.2).
///
/// Covers both built-in and MCP tool names (the latter namespaced
/// `serverName.toolName`, §4) — the parser doesn't distinguish them, since a
/// policy line is just a name-to-decision mapping either way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolPolicy {
    decisions: HashMap<String, ToolDecision>,
}

impl ToolPolicy {
    /// Parses free-form `name: decision` lines (an optional leading
    /// `-`/`*` list marker is stripped first). SOUL.md's Tools policy
    /// section is prose the user wrote, not a strict format: a line that
    /// doesn't match `name: decision`, or whose decision isn't recognized,
    /// is silently skipped rather than rejected — same tolerance 9.1's
    /// section parser applies to the rest of the file.
    pub fn parse(text: &str) -> ToolPolicy {
        let mut decisions = HashMap::new();
        for line in text.lines() {
            let line = line.trim().trim_start_matches(['-', '*']).trim();
            let Some((name, decision)) = line.split_once(':') else {
                continue;
            };
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let decision = match decision.trim().to_lowercase().as_str() {
                "auto" | "auto-run" | "auto run" | "allow" | "allowed" => ToolDecision::Auto,
                "confirm" | "confirmation" | "ask" => ToolDecision::Confirm,
                "off" | "disabled" | "disallow" | "never" => ToolDecision::Off,
                _ => continue,
            };
            decisions.insert(name.to_string(), decision);
        }
        ToolPolicy { decisions }
    }

    /// The effective decision for a tool named `name` registered with
    /// `class`.
    ///
    /// An explicit policy entry always wins. Absent one, a sane default
    /// applies: [`SafetyClass::ReadOnly`] auto-runs, anything higher
    /// requires confirmation — matching EPIC 14.2's "third-party tools
    /// default to confirmation-gated, not auto-run." This method only
    /// decides policy; it never elevates a tool the broker refused to
    /// register in the first place (§14.3), since an unregistered name
    /// never reaches here.
    pub fn decision(&self, name: &str, class: SafetyClass) -> ToolDecision {
        if let Some(decision) = self.decisions.get(name) {
            return *decision;
        }
        match class {
            SafetyClass::ReadOnly => ToolDecision::Auto,
            SafetyClass::SideEffecting | SafetyClass::Dangerous => ToolDecision::Confirm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_six_sections_in_order() {
        let text = r#"
# Identity

You are Marceline, dry and terse.

# Voice

Fast pacing, af_sky voice.

# Values / rules

Never fabricate facts.

# Knowledge about me (the user)

The user's name is Oscar.

# Tools policy

Shell commands require confirmation.

# Examples

User: hi
Marceline: hey.
"#;
        let persona = Persona::parse(text);
        assert_eq!(
            persona.identity.as_deref(),
            Some("You are Marceline, dry and terse.")
        );
        assert_eq!(persona.voice.as_deref(), Some("Fast pacing, af_sky voice."));
        assert_eq!(
            persona.values_rules.as_deref(),
            Some("Never fabricate facts.")
        );
        assert_eq!(
            persona.knowledge.as_deref(),
            Some("The user's name is Oscar.")
        );
        assert_eq!(
            persona.tools_policy.as_deref(),
            Some("Shell commands require confirmation.")
        );
        assert_eq!(
            persona.examples.as_deref(),
            Some("User: hi\nMarceline: hey.")
        );
    }

    #[test]
    fn missing_and_reordered_sections_parse_without_error() {
        let text = r#"
# Voice

Terse.

# Identity

Marceline.
"#;
        let persona = Persona::parse(text);
        assert_eq!(persona.identity.as_deref(), Some("Marceline."));
        assert_eq!(persona.voice.as_deref(), Some("Terse."));
        assert_eq!(persona.values_rules, None);
        assert_eq!(persona.knowledge, None);
        assert_eq!(persona.tools_policy, None);
        assert_eq!(persona.examples, None);
    }

    #[test]
    fn unrecognized_headings_and_preamble_are_ignored() {
        let text = r#"
Some free text before any heading.

# Random Notes

This isn't a recognized section.

# Identity

Marceline.
"#;
        let persona = Persona::parse(text);
        assert_eq!(persona.identity.as_deref(), Some("Marceline."));
        assert_eq!(persona.voice, None);
    }

    #[test]
    fn empty_input_yields_default_persona() {
        assert_eq!(Persona::parse(""), Persona::default());
    }

    #[test]
    fn render_round_trips_canonical_order_regardless_of_source_order() {
        let text = "# Examples\n\nfoo\n\n# Identity\n\nbar\n";
        let persona = Persona::parse(text);
        let rendered = persona.render();
        assert_eq!(rendered, "# Identity\n\nbar\n\n# Examples\n\nfoo");
        let identity_idx = rendered.find("# Identity").unwrap();
        let examples_idx = rendered.find("# Examples").unwrap();
        assert!(identity_idx < examples_idx);
    }

    #[test]
    fn render_omits_missing_sections_with_no_placeholder() {
        let persona = Persona {
            identity: Some("Marceline.".to_string()),
            ..Persona::default()
        };
        assert_eq!(persona.render(), "# Identity\n\nMarceline.");
    }

    #[test]
    fn compiles_into_a_system_prompt_via_render() {
        let persona = Persona::parse("# Identity\n\nYou are Marceline.\n");
        let prompt = crate::llm::prompt::compile_system_prompt(&persona.render(), &[]);
        assert_eq!(prompt, "# Identity\n\nYou are Marceline.");
    }

    #[test]
    fn load_surfaces_io_error_for_missing_file() {
        let err = Persona::load("/nonexistent/path/SOUL.md").unwrap_err();
        assert!(matches!(err, SoulError::Io { .. }));
    }

    #[test]
    fn tool_policy_parses_explicit_decisions() {
        let policy = ToolPolicy::parse(
            "- shell.run: confirm\n* web_search: auto\ndelete_file: off\nnot a policy line\n",
        );
        assert_eq!(
            policy.decision("shell.run", SafetyClass::ReadOnly),
            ToolDecision::Confirm
        );
        assert_eq!(
            policy.decision("web_search", SafetyClass::SideEffecting),
            ToolDecision::Auto
        );
        assert_eq!(
            policy.decision("delete_file", SafetyClass::ReadOnly),
            ToolDecision::Off
        );
    }

    #[test]
    fn tool_policy_falls_back_to_class_default_when_unlisted() {
        let policy = ToolPolicy::default();
        assert_eq!(
            policy.decision("get_time", SafetyClass::ReadOnly),
            ToolDecision::Auto
        );
        assert_eq!(
            policy.decision("send_email", SafetyClass::SideEffecting),
            ToolDecision::Confirm
        );
        assert_eq!(
            policy.decision("rm_rf", SafetyClass::Dangerous),
            ToolDecision::Confirm
        );
    }

    #[test]
    fn persona_tool_policy_reads_the_tools_policy_section() {
        let persona = Persona::parse("# Tools policy\n\nshell.run: confirm\n");
        let policy = persona.tool_policy();
        assert_eq!(
            policy.decision("shell.run", SafetyClass::ReadOnly),
            ToolDecision::Confirm
        );
    }
}
