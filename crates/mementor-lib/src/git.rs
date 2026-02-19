use std::path::{Path, PathBuf};

/// The result of resolving a working directory's git worktree status.
///
/// Returned by [`resolve_worktree`]. Classifies the working directory as being
/// inside a primary worktree, a linked worktree, or not inside any git
/// repository at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedWorktree {
    /// The working directory is inside a primary (non-linked) worktree.
    /// Contains the primary worktree root (where `.git` is a directory).
    Primary(PathBuf),
    /// The working directory is inside a linked worktree.
    /// Contains the resolved primary worktree root.
    Linked(PathBuf),
    /// The working directory is not inside any git repository.
    NotGitRepo,
}

impl ResolvedWorktree {
    /// Returns the primary worktree root, regardless of whether this is a
    /// primary or linked worktree. Returns `None` if not in a git repo.
    pub fn primary_root(&self) -> Option<&Path> {
        match self {
            Self::Primary(root) | Self::Linked(root) => Some(root),
            Self::NotGitRepo => None,
        }
    }

    /// Returns `true` if this is a linked (non-primary) worktree.
    pub fn is_linked(&self) -> bool {
        matches!(self, Self::Linked(_))
    }
}

/// Classify the git worktree that contains `cwd` and resolve its primary root.
///
/// Walks up from `cwd` toward the filesystem root, looking for a `.git` entry:
///
/// - **Directory** (`.git/`): this is the primary worktree root.
///   Returns [`ResolvedWorktree::Primary`].
/// - **File** (`.git`): could be a linked worktree or a submodule.
///   - If `<gitdir>/commondir` exists, this is a **linked worktree**. Follow
///     the `commondir` chain to resolve the common `.git` directory and return
///     [`ResolvedWorktree::Linked`] with the primary worktree root.
///   - If `commondir` is absent, this is a **submodule**. Skip it and keep
///     walking up — the submodule is part of the larger project.
///
/// Returns [`ResolvedWorktree::NotGitRepo`] if no `.git` entry is found at any
/// ancestor, or if an I/O or parse error occurs.
pub fn resolve_worktree(cwd: &Path) -> ResolvedWorktree {
    let mut current = cwd.to_path_buf();

    loop {
        let git_entry = current.join(".git");

        if git_entry.is_dir() {
            return ResolvedWorktree::Primary(current);
        }

        if git_entry.is_file()
            && let Some(root) = try_resolve_linked_worktree(&current, &git_entry)
        {
            return ResolvedWorktree::Linked(root);
        }
        // If .git is a file without commondir (submodule), skip and keep
        // walking up.

        if !current.pop() {
            return ResolvedWorktree::NotGitRepo;
        }
    }
}

/// Try to resolve a linked worktree's `.git` file to the primary root.
///
/// Returns `Some(primary_root)` if this is a linked worktree (i.e.,
/// `commondir` exists inside the gitdir). Returns `None` if it's a submodule
/// (no `commondir`) or on any I/O/parse error.
fn try_resolve_linked_worktree(dir: &Path, git_file: &Path) -> Option<PathBuf> {
    let content = std::fs::read_to_string(git_file).ok()?;
    let gitdir_ref = content.strip_prefix("gitdir: ")?.trim();

    let gitdir = if Path::new(gitdir_ref).is_absolute() {
        PathBuf::from(gitdir_ref)
    } else {
        dir.join(gitdir_ref)
    };

    // Only linked worktrees have a commondir file. Submodules do not.
    let commondir_file = gitdir.join("commondir");
    if !commondir_file.is_file() {
        return None;
    }

    let commondir_ref = std::fs::read_to_string(&commondir_file).ok()?;
    let commondir_ref = commondir_ref.trim();

    let common_git_dir = if Path::new(commondir_ref).is_absolute() {
        PathBuf::from(commondir_ref)
    } else {
        gitdir.join(commondir_ref)
    };

    let common_git_dir = std::fs::canonicalize(&common_git_dir).ok()?;
    let primary_root = common_git_dir.parent()?;

    Some(primary_root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use mementor_test_util::git::{assert_paths_eq, init_git_repo, run_git};

    use super::*;

    #[test]
    fn resolve_worktree_primary_at_root() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        init_git_repo(&repo);

        let result = resolve_worktree(&repo);
        assert!(matches!(result, ResolvedWorktree::Primary(_)));
        assert!(!result.is_linked());
        assert_paths_eq(result.primary_root().unwrap(), &repo);
    }

    #[test]
    fn resolve_worktree_primary_from_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        init_git_repo(&repo);

        let subdir = repo.join("src").join("deep");
        std::fs::create_dir_all(&subdir).unwrap();

        let result = resolve_worktree(&subdir);
        assert!(matches!(result, ResolvedWorktree::Primary(_)));
        assert!(!result.is_linked());
    }

    #[test]
    fn resolve_worktree_linked() {
        let tmp = tempfile::tempdir().unwrap();
        let main_dir = tmp.path().join("main");
        std::fs::create_dir_all(&main_dir).unwrap();
        init_git_repo(&main_dir);

        let wt_dir = tmp.path().join("wt");
        run_git(
            &main_dir,
            &["worktree", "add", wt_dir.to_str().unwrap(), "-b", "test"],
        );

        let result = resolve_worktree(&wt_dir);
        assert!(matches!(result, ResolvedWorktree::Linked(_)));
        assert!(result.is_linked());
        assert_paths_eq(result.primary_root().unwrap(), &main_dir);
    }

    #[test]
    fn resolve_worktree_linked_from_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let main_dir = tmp.path().join("main");
        std::fs::create_dir_all(&main_dir).unwrap();
        init_git_repo(&main_dir);

        let wt_dir = tmp.path().join("wt");
        run_git(
            &main_dir,
            &["worktree", "add", wt_dir.to_str().unwrap(), "-b", "test"],
        );

        let subdir = wt_dir.join("src").join("deep");
        std::fs::create_dir_all(&subdir).unwrap();

        let result = resolve_worktree(&subdir);
        assert!(matches!(result, ResolvedWorktree::Linked(_)));
        assert!(result.is_linked());
    }

    #[test]
    fn resolve_worktree_not_git_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_worktree(tmp.path());
        assert!(matches!(result, ResolvedWorktree::NotGitRepo));
        assert!(result.primary_root().is_none());
        assert!(!result.is_linked());
    }

    // ---------------------------------------------------------------
    // Submodule cases
    // ---------------------------------------------------------------

    /// Helper: create a repo, add a submodule, return (parent_dir, sub_dir).
    fn setup_repo_with_submodule(tmp: &Path) -> (PathBuf, PathBuf) {
        // Create the "remote" repo that will become the submodule.
        let remote = tmp.join("remote-sub");
        std::fs::create_dir_all(&remote).unwrap();
        init_git_repo(&remote);

        // Create the parent repo.
        let parent = tmp.join("parent");
        std::fs::create_dir_all(&parent).unwrap();
        init_git_repo(&parent);

        // Add the submodule.
        run_git(
            &parent,
            &["submodule", "add", remote.to_str().unwrap(), "sub"],
        );
        run_git(&parent, &["commit", "-m", "add submodule"]);

        let sub_dir = parent.join("sub");
        (parent, sub_dir)
    }

    #[test]
    fn submodule_root_skips_to_parent_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let (parent, sub_dir) = setup_repo_with_submodule(tmp.path());

        let result = resolve_worktree(&sub_dir);
        assert!(matches!(result, ResolvedWorktree::Primary(_)));
        assert_paths_eq(result.primary_root().unwrap(), &parent);
    }

    #[test]
    fn submodule_subdirectory_skips_to_parent_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let (parent, sub_dir) = setup_repo_with_submodule(tmp.path());

        let deep = sub_dir.join("src").join("lib");
        std::fs::create_dir_all(&deep).unwrap();

        let result = resolve_worktree(&deep);
        assert!(matches!(result, ResolvedWorktree::Primary(_)));
        assert_paths_eq(result.primary_root().unwrap(), &parent);
    }

    #[test]
    fn nested_submodule_skips_to_top_repo() {
        let tmp = tempfile::tempdir().unwrap();

        // Level 0: top repo
        let top = tmp.path().join("top");
        std::fs::create_dir_all(&top).unwrap();
        init_git_repo(&top);

        // Level 1: sub-a (remote for submodule)
        let remote_a = tmp.path().join("remote-a");
        std::fs::create_dir_all(&remote_a).unwrap();
        init_git_repo(&remote_a);

        // Level 2: sub-b (remote for nested submodule inside sub-a)
        let remote_b = tmp.path().join("remote-b");
        std::fs::create_dir_all(&remote_b).unwrap();
        init_git_repo(&remote_b);

        // Add sub-b as submodule of remote-a.
        run_git(
            &remote_a,
            &["submodule", "add", remote_b.to_str().unwrap(), "sub-b"],
        );
        run_git(&remote_a, &["commit", "-m", "add sub-b"]);

        // Add remote-a as submodule of top.
        run_git(
            &top,
            &["submodule", "add", remote_a.to_str().unwrap(), "sub-a"],
        );
        run_git(&top, &["commit", "-m", "add sub-a"]);

        // Initialize nested submodule.
        run_git(&top, &["submodule", "update", "--init", "--recursive"]);

        let nested = top.join("sub-a").join("sub-b");
        let result = resolve_worktree(&nested);
        assert!(matches!(result, ResolvedWorktree::Primary(_)));
        assert_paths_eq(result.primary_root().unwrap(), &top);
    }

    #[test]
    fn submodule_in_linked_worktree_resolves_to_primary_root() {
        let tmp = tempfile::tempdir().unwrap();

        // Create a remote repo for the submodule.
        let remote = tmp.path().join("remote-sub");
        std::fs::create_dir_all(&remote).unwrap();
        init_git_repo(&remote);

        // Create the main repo with a submodule.
        let main_dir = tmp.path().join("main");
        std::fs::create_dir_all(&main_dir).unwrap();
        init_git_repo(&main_dir);
        run_git(
            &main_dir,
            &["submodule", "add", remote.to_str().unwrap(), "sub"],
        );
        run_git(&main_dir, &["commit", "-m", "add submodule"]);

        // Create a linked worktree.
        let wt_dir = tmp.path().join("worktree");
        run_git(
            &main_dir,
            &["worktree", "add", wt_dir.to_str().unwrap(), "-b", "wt-test"],
        );

        // Initialize submodule in the worktree.
        run_git(&wt_dir, &["submodule", "update", "--init"]);

        let sub_in_wt = wt_dir.join("sub");
        let result = resolve_worktree(&sub_in_wt);
        assert!(matches!(result, ResolvedWorktree::Linked(_)));
        assert_paths_eq(result.primary_root().unwrap(), &main_dir);
    }
}
