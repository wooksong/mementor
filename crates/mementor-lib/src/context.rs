use std::path::{Path, PathBuf};

/// Abstracts environment and configuration for dependency injection.
pub trait MementorContext: Clone {
    /// Root directory of the project where mementor is enabled.
    fn project_root(&self) -> &Path;

    /// Path to the mementor `SQLite` database file (`mementor.db`).
    /// Default: `<project_root>/.mementor/mementor.db`
    fn db_path(&self) -> PathBuf {
        self.mementor_dir().join("mementor.db")
    }

    /// Path to the `.mementor/` directory.
    /// Default: `<project_root>/.mementor/`
    fn mementor_dir(&self) -> PathBuf {
        self.project_root().join(".mementor")
    }

    /// Path to the Claude settings file.
    /// Default: `<project_root>/.claude/settings.json`
    fn claude_settings_path(&self) -> PathBuf {
        self.project_root().join(".claude").join("settings.json")
    }

    /// Path to the project's `.gitignore` file.
    /// Default: `<project_root>/.gitignore`
    fn gitignore_path(&self) -> PathBuf {
        self.project_root().join(".gitignore")
    }
}

/// Real implementation that uses an actual filesystem path.
#[derive(Clone, Debug)]
pub struct RealMementorContext {
    project_root: PathBuf,
}

impl RealMementorContext {
    /// Create a new context rooted at the given path.
    #[must_use]
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    /// Create a context from the current working directory.
    pub fn from_cwd() -> anyhow::Result<Self> {
        let cwd = std::env::current_dir()?;
        Ok(Self::new(cwd))
    }
}

impl MementorContext for RealMementorContext {
    fn project_root(&self) -> &Path {
        &self.project_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_path_is_under_mementor_dir() {
        let ctx = RealMementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(
            ctx.db_path(),
            PathBuf::from("/tmp/project/.mementor/mementor.db")
        );
    }

    #[test]
    fn mementor_dir_is_under_project_root() {
        let ctx = RealMementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(ctx.mementor_dir(), PathBuf::from("/tmp/project/.mementor"));
    }

    #[test]
    fn claude_settings_path() {
        let ctx = RealMementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(
            ctx.claude_settings_path(),
            PathBuf::from("/tmp/project/.claude/settings.json")
        );
    }

    #[test]
    fn gitignore_path() {
        let ctx = RealMementorContext::new(PathBuf::from("/tmp/project"));
        assert_eq!(
            ctx.gitignore_path(),
            PathBuf::from("/tmp/project/.gitignore")
        );
    }
}
