use serde::Deserialize;

/// Input received from the Claude Code Stop hook via stdin.
#[derive(Debug, Deserialize)]
pub struct StopHookInput {
    /// The session ID of the Claude Code conversation.
    #[serde(rename = "sessionId")]
    pub session_id: String,
    /// Path to the JSONL transcript file.
    #[serde(rename = "transcriptPath")]
    pub transcript_path: String,
    /// The project working directory.
    pub cwd: String,
}

/// Input received from the Claude Code `UserPromptSubmit` hook via stdin.
#[derive(Debug, Deserialize)]
pub struct PromptHookInput {
    /// The session ID of the Claude Code conversation.
    #[serde(rename = "sessionId")]
    pub session_id: String,
    /// The user's prompt text.
    pub prompt: String,
    /// The project working directory.
    pub cwd: String,
}

/// Read and parse the stop hook input from stdin.
pub fn read_stop_input(reader: &mut dyn std::io::Read) -> anyhow::Result<StopHookInput> {
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let input: StopHookInput = serde_json::from_str(&buf)?;
    Ok(input)
}

/// Read and parse the prompt hook input from stdin.
pub fn read_prompt_input(reader: &mut dyn std::io::Read) -> anyhow::Result<PromptHookInput> {
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let input: PromptHookInput = serde_json::from_str(&buf)?;
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stop_input() {
        let json = r#"{"sessionId": "abc-123", "transcriptPath": "/tmp/transcript.jsonl", "cwd": "/home/user/project"}"#;
        let input = read_stop_input(&mut json.as_bytes()).unwrap();
        assert_eq!(input.session_id, "abc-123");
        assert_eq!(input.transcript_path, "/tmp/transcript.jsonl");
        assert_eq!(input.cwd, "/home/user/project");
    }

    #[test]
    fn parse_prompt_input() {
        let json = r#"{"sessionId": "abc-123", "prompt": "How do I fix the bug?", "cwd": "/home/user/project"}"#;
        let input = read_prompt_input(&mut json.as_bytes()).unwrap();
        assert_eq!(input.session_id, "abc-123");
        assert_eq!(input.prompt, "How do I fix the bug?");
    }

    #[test]
    fn missing_field_errors() {
        let json = r#"{"sessionId": "abc"}"#;
        let result = read_stop_input(&mut json.as_bytes());
        assert!(result.is_err());
    }
}
