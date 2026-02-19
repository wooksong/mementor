use std::path::{Path, PathBuf};

/// Environment and configuration for a mementor-enabled project.
#[derive(Clone, Debug)]
pub struct MementorContext {
    /// The resolved primary worktree root (used for DB, settings, etc.).
    project_root: PathBuf,
    /// The actual working directory (may differ from `project_root` in a
    /// linked worktree or subdirectory).
    cwd: PathBuf,
    /// `true` when `cwd` is inside a linked (non-primary) git worktree.
    is_linked_worktree: bool,
    log_dir: Option<PathBuf>,
}

impl MementorContext {
    /// Create a new context rooted at the given path (no log directory).
    ///
    /// Sets `cwd` equal to `project_root`. Use [`Self::with_cwd_and_log_dir`]
    /// when the actual working directory differs from the project root.
    #[must_use]
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            cwd: project_root.clone(),
            project_root,
            is_linked_worktree: false,
            log_dir: None,
        }
    }

    /// Create a new context with an explicit log directory.
    ///
    /// Sets `cwd` equal to `project_root`.
    #[must_use]
    pub fn with_log_dir(project_root: PathBuf, log_dir: Option<PathBuf>) -> Self {
        Self {
            cwd: project_root.clone(),
            project_root,
            is_linked_worktree: false,
            log_dir,
        }
    }

    /// Create a new context with separate cwd and project root.
    ///
    /// Use this when the actual working directory (e.g., a linked worktree or
    /// subdirectory) differs from the resolved primary worktree root.
    #[must_use]
    pub fn with_cwd_and_log_dir(
        cwd: PathBuf,
        project_root: PathBuf,
        is_linked_worktree: bool,
        log_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            project_root,
            cwd,
            is_linked_worktree,
            log_dir,
        }
    }

    /// Create a context from the current working directory (no log directory).
    pub fn from_cwd() -> anyhow::Result<Self> {
        let cwd = std::env::current_dir()?;
        Ok(Self::new(cwd))
    }

    /// Root directory of the project where mementor is enabled.
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    /// The actual working directory at startup.
    ///
    /// May be a linked worktree or subdirectory that differs from
    /// [`Self::project_root`].
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Returns `true` if the working directory is inside a linked (non-primary)
    /// git worktree.
    ///
    /// Determined at startup by [`crate::git::resolve_worktree`] based on the
    /// `.git` entry type — not by path comparison, so symlinks do not affect
    /// the result.
    pub fn is_linked_worktree(&self) -> bool {
        self.is_linked_worktree
    }

    /// Optional parent directory for log file output.
    /// When set, operational logs are written to JSONL files under this path.
    pub fn log_dir(&self) -> Option<&Path> {
        self.log_dir.as_deref()
    }

    /// Path to the mementor `SQLite` database file (`mementor.db`).
    /// Default: `<project_root>/.mementor/mementor.db`
    pub fn db_path(&self) -> PathBuf {
        self.mementor_dir().join("mementor.db")
    }

    /// Path to the `.mementor/` directory.
    /// Default: `<project_root>/.mementor/`
    pub fn mementor_dir(&self) -> PathBuf {
        self.project_root.join(".mementor")
    }

    /// Path to the Claude settings file.
    /// Default: `<project_root>/.claude/settings.json`
    pub fn claude_settings_path(&self) -> PathBuf {
        self.project_root.join(".claude").join("settings.json")
    }

    /// Path to the project's `.gitignore` file.
    /// Default: `<project_root>/.gitignore`
    pub fn gitignore_path(&self) -> PathBuf {
        self.project_root.join(".gitignore")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_path_is_under_mementor_dir() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(
            ctx.db_path(),
            PathBuf::from("/tmp/project/.mementor/mementor.db")
        );
    }

    #[test]
    fn mementor_dir_is_under_project_root() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(ctx.mementor_dir(), PathBuf::from("/tmp/project/.mementor"));
    }

    #[test]
    fn claude_settings_path() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(
            ctx.claude_settings_path(),
            PathBuf::from("/tmp/project/.claude/settings.json")
        );
    }

    #[test]
    fn gitignore_path() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(
            ctx.gitignore_path(),
            PathBuf::from("/tmp/project/.gitignore")
        );
    }

    #[test]
    fn log_dir_defaults_to_none() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert!(ctx.log_dir().is_none());
    }

    #[test]
    fn log_dir_with_explicit_value() {
        let ctx = MementorContext::with_log_dir(
            PathBuf::from("/tmp/project"),
            Some(PathBuf::from("/tmp/logs")),
        );
        assert_eq!(ctx.log_dir(), Some(Path::new("/tmp/logs")));
    }

    #[test]
    fn cwd_equals_project_root_by_default() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(ctx.cwd(), ctx.project_root());
    }

    #[test]
    fn cwd_can_differ_from_project_root() {
        let ctx = MementorContext::with_cwd_and_log_dir(
            PathBuf::from("/tmp/worktree"),
            PathBuf::from("/tmp/project"),
            true,
            None,
        );
        assert_eq!(ctx.cwd(), Path::new("/tmp/worktree"));
        assert_eq!(ctx.project_root(), Path::new("/tmp/project"));
        assert!(ctx.is_linked_worktree());
        // DB path still derives from project_root, not cwd.
        assert_eq!(
            ctx.db_path(),
            PathBuf::from("/tmp/project/.mementor/mementor.db"),
        );
    }

    #[test]
    fn is_linked_worktree_defaults_to_false() {
        let ctx = MementorContext::new(PathBuf::from("/tmp/project"));
        assert!(!ctx.is_linked_worktree());
    }
}
