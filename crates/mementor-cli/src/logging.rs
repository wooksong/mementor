use std::fs::{self, OpenOptions};
use std::path::Path;
use std::sync::Mutex;

use sha2::{Digest, Sha256};
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use mementor_lib::context::MementorContext;

/// Initialize file-based JSONL logging if the context has a log directory.
///
/// When `ctx.log_dir()` is `None`, this is a no-op.
/// All errors during initialization are silently ignored so that logging
/// failures never prevent mementor from functioning.
pub fn init_file_logging(ctx: &MementorContext) {
    let Some(log_dir) = ctx.log_dir() else {
        return;
    };
    let _ = try_init_file_logging(log_dir, ctx.project_root());
}

fn try_init_file_logging(log_dir: &Path, project_root: &Path) -> anyhow::Result<()> {
    let (dirname, digest) = log_subdir_parts(project_root);
    let subdir_name = format!("{dirname}-{digest}");
    let log_subdir = log_dir.join(&subdir_name);
    fs::create_dir_all(&log_subdir)?;

    let today = jiff::Zoned::now().date();
    let filename = format!("{today}.jsonl");
    let file_path = log_subdir.join(&filename);

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;

    tracing_subscriber::registry()
        .with(fmt::layer().json().with_writer(Mutex::new(file)))
        .init();

    cleanup_old_logs(&log_subdir, today);

    Ok(())
}

/// Compute the subdirectory name parts from a project root path.
///
/// Returns `(dirname, digest)` where:
/// - `dirname` is the basename of the project root
/// - `digest` is the first 8 hex characters of the SHA-256 of the absolute path
pub fn log_subdir_parts(project_root: &Path) -> (String, String) {
    let dirname = project_root.file_name().map_or_else(
        || "unknown".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );

    let mut hasher = Sha256::new();
    hasher.update(project_root.to_string_lossy().as_bytes());
    let hash = hasher.finalize();
    let digest = hash.iter().take(4).fold(String::new(), |mut acc, b| {
        use std::fmt::Write;
        let _ = write!(acc, "{b:02x}");
        acc
    });

    (dirname, digest)
}

/// Delete log files older than 28 days. Errors are silently ignored.
fn cleanup_old_logs(log_subdir: &Path, today: jiff::civil::Date) {
    let Ok(entries) = fs::read_dir(log_subdir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext != "jsonl" {
            continue;
        }
        let Ok(file_date) = stem.parse::<jiff::civil::Date>() else {
            continue;
        };
        if let Ok(span) = today.since(file_date)
            && span.get_days() >= 28
        {
            let _ = fs::remove_file(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn log_subdir_parts_computes_dirname_and_digest() {
        let (dirname, digest) = log_subdir_parts(Path::new("/Users/test/my-project"));
        assert_eq!(dirname, "my-project");
        assert_eq!(digest.len(), 8);
        // Digest should be deterministic
        let (_, digest2) = log_subdir_parts(Path::new("/Users/test/my-project"));
        assert_eq!(digest, digest2);
    }

    #[test]
    fn log_subdir_parts_different_paths_produce_different_digests() {
        let (_, d1) = log_subdir_parts(Path::new("/a/project"));
        let (_, d2) = log_subdir_parts(Path::new("/b/project"));
        assert_ne!(d1, d2);
    }

    #[test]
    fn cleanup_old_logs_removes_old_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // Create an old log file (30 days ago)
        let old_date = jiff::Zoned::now()
            .date()
            .checked_sub(jiff::Span::new().days(30))
            .unwrap();
        let old_file = dir.join(format!("{old_date}.jsonl"));
        fs::write(&old_file, "{}").unwrap();

        // Create a recent log file (1 day ago)
        let recent_date = jiff::Zoned::now()
            .date()
            .checked_sub(jiff::Span::new().days(1))
            .unwrap();
        let recent_file = dir.join(format!("{recent_date}.jsonl"));
        fs::write(&recent_file, "{}").unwrap();

        let today = jiff::Zoned::now().date();
        cleanup_old_logs(dir, today);

        assert!(!old_file.exists(), "old log file should be deleted");
        assert!(recent_file.exists(), "recent log file should be kept");
    }

    #[test]
    fn cleanup_ignores_non_jsonl_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        let other_file = dir.join("notes.txt");
        fs::write(&other_file, "keep me").unwrap();

        let today = jiff::Zoned::now().date();
        cleanup_old_logs(dir, today);

        assert!(other_file.exists(), "non-jsonl files should not be deleted");
    }

    #[test]
    fn log_subdir_parts_handles_root_path() {
        let (dirname, digest) = log_subdir_parts(&PathBuf::from("/"));
        // Root path has no file_name, falls back to "unknown"
        assert_eq!(dirname, "unknown");
        assert_eq!(digest.len(), 8);
    }
}
