//! Shared path confinement for filesystem tools (`read_file`, `list_dir`;
//! SPEC.md §4, §5.1, EPIC 6.2, issue #185).
//!
//! `SafetyClass::ReadOnly` describes the effect on the filesystem, not the
//! effect on the user: an unconfined "read any file" tool auto-run by
//! policy can reach `~/.ssh/id_rsa`, `.env` files, or the conversation
//! history database, and a tool result becomes part of the prompt the
//! moment it's returned. [`Sandbox`] is what keeps a read confined to an
//! explicit root, resistant to `..` traversal and symlink escapes
//! (canonicalizing resolves both), with a denylist for sensitive names
//! that might live inside an otherwise-fine root.

use std::path::{Path, PathBuf};

/// Names denied anywhere under the sandbox root, regardless of the
/// requested path — a root that happens to contain a `.ssh` directory or
/// an `.env` file stays unreadable even though it's inside the root.
/// Matched case-insensitively against each path component.
const DENIED_NAMES: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".git",
    ".env",
    ".netrc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials",
    "history.db",
];

/// Confines filesystem tool access to a fixed root directory.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// The root's canonical (symlink-resolved) form, computed once at
    /// construction so every [`resolve`][Sandbox::resolve] call reuses it
    /// rather than re-resolving the root on every request.
    canonical_root: PathBuf,
}

impl Sandbox {
    /// Builds a sandbox rooted at `root`. Fails if `root` doesn't exist or
    /// can't be canonicalized — an unreadable root would silently confine
    /// every request to nothing, which should surface at startup, not on
    /// the first tool call.
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            canonical_root: root.as_ref().canonicalize()?,
        })
    }

    /// The root every request is confined to, canonical form.
    pub fn root(&self) -> &Path {
        &self.canonical_root
    }

    /// Resolves `requested` (as given by the model) to a real path inside
    /// the sandbox, or a human-readable reason it was refused.
    ///
    /// `requested` must be relative — an absolute path would otherwise
    /// override the root entirely via [`Path::join`]'s own semantics, so
    /// it's rejected before that can happen. The joined path is then
    /// canonicalized, which collapses any `..` and follows symlinks to
    /// their real target, and the result must still be inside the root:
    /// that's what makes both traversal and a symlink pointing outside
    /// the sandbox fail the same check rather than needing separate ones.
    pub fn resolve(&self, requested: &str) -> Result<PathBuf, String> {
        let requested_path = Path::new(requested);
        if requested_path.is_absolute() {
            return Err(format!(
                "{requested} is outside the allowed directory ({}): absolute paths aren't permitted",
                self.canonical_root.display()
            ));
        }

        let joined = self.canonical_root.join(requested_path);
        let resolved = joined.canonicalize().map_err(|err| format!("failed to resolve {requested}: {err}"))?;

        if !resolved.starts_with(&self.canonical_root) {
            return Err(format!(
                "{requested} is outside the allowed directory ({})",
                self.canonical_root.display()
            ));
        }

        if let Ok(relative) = resolved.strip_prefix(&self.canonical_root) {
            for component in relative.components() {
                let name = component.as_os_str().to_string_lossy().to_lowercase();
                if DENIED_NAMES.iter().any(|denied| name == *denied) {
                    return Err(format!("{requested} is denied: {name} is not readable"));
                }
            }
        }

        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox_in(dir: &Path) -> Sandbox {
        Sandbox::new(dir).expect("root exists")
    }

    #[test]
    fn a_relative_path_inside_the_root_resolves() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        let sandbox = sandbox_in(dir.path());

        let resolved = sandbox.resolve("a.txt").unwrap();
        assert_eq!(resolved, dir.path().canonicalize().unwrap().join("a.txt"));
    }

    #[test]
    fn an_absolute_path_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());

        assert!(sandbox.resolve("/etc/passwd").is_err());
    }

    #[test]
    fn a_dotdot_escape_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let sandbox = sandbox_in(dir.path());

        let escape = format!("../{}", outside.path().file_name().unwrap().to_str().unwrap());
        assert!(sandbox.resolve(&escape).is_err());
    }

    #[test]
    fn a_symlink_pointing_outside_the_root_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, b"top secret").unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&secret, &link).unwrap();

        let sandbox = sandbox_in(dir.path());
        assert!(sandbox.resolve("link").is_err());
    }

    #[test]
    fn a_denied_name_inside_the_root_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".ssh")).unwrap();
        std::fs::write(dir.path().join(".ssh/id_rsa"), b"private").unwrap();
        let sandbox = sandbox_in(dir.path());

        assert!(sandbox.resolve(".ssh/id_rsa").is_err());
    }

    #[test]
    fn a_missing_path_reports_the_original_request() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());

        let err = sandbox.resolve("nope.txt").unwrap_err();
        assert!(err.contains("nope.txt"));
    }
}
