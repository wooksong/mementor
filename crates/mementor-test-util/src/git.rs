use std::path::Path;
use std::process::Command;

/// Initialize a git repository with a test user config and an initial empty
/// commit.
pub fn init_git_repo(dir: &Path) {
    run_git(dir, &["init"]);
    run_git(dir, &["config", "user.email", "test@test.com"]);
    run_git(dir, &["config", "user.name", "Test"]);
    run_git(dir, &["commit", "--allow-empty", "-m", "initial"]);
}

/// Run a git command in the given directory.
///
/// Passes `-c protocol.file.allow=always` to allow file-protocol operations
/// (required for submodule clones in tests on Git 2.38.1+).
///
/// # Panics
///
/// Panics if the command fails to start or exits with a non-zero status.
pub fn run_git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(["-c", "protocol.file.allow=always"])
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git command failed to start");
    assert!(
        output.status.success(),
        "git {} failed in {}: {}",
        args.join(" "),
        dir.display(),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Assert that two paths are equal after canonicalization.
///
/// Resolves symlinks before comparison, avoiding false negatives from macOS
/// `/private/tmp` vs `/tmp` aliasing.
///
/// # Panics
///
/// Panics if either path cannot be canonicalized or if the paths differ.
pub fn assert_paths_eq(actual: &Path, expected: &Path) {
    let actual = std::fs::canonicalize(actual)
        .unwrap_or_else(|_| panic!("cannot canonicalize {}", actual.display()));
    let expected = std::fs::canonicalize(expected)
        .unwrap_or_else(|_| panic!("cannot canonicalize {}", expected.display()));
    assert_eq!(actual, expected);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_git_repo_creates_valid_repo() {
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path());

        let output = Command::new("git")
            .args(["log", "--oneline"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("initial"),
            "expected initial commit in git log",
        );
    }

    #[test]
    #[should_panic(expected = "git")]
    fn run_git_panics_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        run_git(tmp.path(), &["log"]);
    }

    #[test]
    fn assert_paths_eq_resolves_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        assert_paths_eq(tmp.path(), tmp.path());
    }
}
