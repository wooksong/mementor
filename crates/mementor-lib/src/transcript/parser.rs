use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::Context;
use tracing::warn;

use super::types::TranscriptEntry;

/// Parsed transcript messages with their line indices.
#[derive(Debug)]
pub struct ParsedMessage {
    /// 0-based line index in the JSONL file.
    pub line_index: usize,
    /// Role: "user" or "assistant".
    pub role: String,
    /// Extracted text content (`tool_use`/`tool_result` blocks stripped).
    pub text: String,
}

/// Read transcript JSONL file starting from `start_line` (0-based).
/// Returns parsed messages with user/assistant text content only.
pub fn parse_transcript(path: &Path, start_line: usize) -> anyhow::Result<Vec<ParsedMessage>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open transcript: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut messages = Vec::new();

    for (line_idx, line_result) in reader.lines().enumerate() {
        if line_idx < start_line {
            continue;
        }

        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                warn!(line = line_idx, error = %e, "Failed to read line, skipping");
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let entry: TranscriptEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                warn!(line = line_idx, error = %e, "Failed to parse JSON line, skipping");
                continue;
            }
        };

        let Some(message) = entry.message else {
            continue;
        };

        let role = message.role.as_str();
        if role != "user" && role != "assistant" {
            continue;
        }

        let text = message.content.extract_text();
        if text.trim().is_empty() {
            continue;
        }

        messages.push(ParsedMessage {
            line_index: line_idx,
            role: message.role,
            text,
        });
    }

    Ok(messages)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    fn write_jsonl(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_full_transcript() {
        let f = write_jsonl(&[
            r#"{"type":"user","message":{"role":"user","content":"Hello"}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Hi there"}}"#,
        ]);

        let msgs = parse_transcript(f.path(), 0).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].text, "Hello");
        assert_eq!(msgs[0].line_index, 0);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].text, "Hi there");
        assert_eq!(msgs[1].line_index, 1);
    }

    #[test]
    fn parse_from_offset() {
        let f = write_jsonl(&[
            r#"{"type":"user","message":{"role":"user","content":"First"}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Second"}}"#,
            r#"{"type":"user","message":{"role":"user","content":"Third"}}"#,
        ]);

        let msgs = parse_transcript(f.path(), 2).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text, "Third");
        assert_eq!(msgs[0].line_index, 2);
    }

    #[test]
    fn skip_malformed_lines() {
        let f = write_jsonl(&[
            r#"{"type":"user","message":{"role":"user","content":"Good"}}"#,
            r#"not valid json"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Also good"}}"#,
        ]);

        let msgs = parse_transcript(f.path(), 0).unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn skip_empty_content() {
        let f = write_jsonl(&[
            r#"{"type":"user","message":{"role":"user","content":""}}"#,
            r#"{"type":"user","message":{"role":"user","content":"Real content"}}"#,
        ]);

        let msgs = parse_transcript(f.path(), 0).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text, "Real content");
    }

    #[test]
    fn empty_file() {
        let f = write_jsonl(&[]);
        let msgs = parse_transcript(f.path(), 0).unwrap();
        assert!(msgs.is_empty());
    }

    #[test]
    fn skip_entries_without_message() {
        let f = write_jsonl(&[
            r#"{"type":"system","uuid":"abc"}"#,
            r#"{"type":"user","message":{"role":"user","content":"Hello"}}"#,
        ]);

        let msgs = parse_transcript(f.path(), 0).unwrap();
        assert_eq!(msgs.len(), 1);
    }
}
