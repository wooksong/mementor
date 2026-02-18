use serde::Deserialize;

/// A single entry (line) in the Claude Code transcript JSONL file.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptEntry {
    #[serde(rename = "type")]
    pub entry_type: Option<String>,
    pub uuid: Option<String>,
    pub session_id: Option<String>,
    pub timestamp: Option<String>,
    pub message: Option<Message>,
}

/// A message within a transcript entry.
#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

/// Content can be a plain string or an array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A single content block within a message's content array.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: Option<String>,
        #[allow(dead_code)]
        signature: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        #[allow(dead_code)]
        id: Option<String>,
        #[allow(dead_code)]
        name: Option<String>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[allow(dead_code)]
        tool_use_id: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

impl Content {
    /// Extract all text content from `text` and `thinking` blocks.
    /// Skips `tool_use`, `tool_result`, and unknown block types.
    pub fn extract_text(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Blocks(blocks) => {
                let texts: Vec<&str> = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::Thinking { thinking, .. } => {
                            thinking.as_deref().filter(|s| !s.is_empty())
                        }
                        ContentBlock::ToolUse { .. }
                        | ContentBlock::ToolResult { .. }
                        | ContentBlock::Unknown => None,
                    })
                    .collect();
                texts.join("\n\n")
            }
        }
    }

    /// Returns `true` if any content block deserialized as `Unknown`.
    pub fn has_unknown_blocks(&self) -> bool {
        match self {
            Content::Text(_) => false,
            Content::Blocks(blocks) => blocks.iter().any(|b| matches!(b, ContentBlock::Unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_text_content() {
        let json = r#"{"role": "user", "content": "Hello world"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.extract_text(), "Hello world");
    }

    #[test]
    fn deserialize_blocks_content() {
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me analyze this"},
                {"type": "text", "text": "Here is the code:"},
                {"type": "tool_use", "id": "t1", "name": "write"},
                {"type": "text", "text": "Done."}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(
            msg.content.extract_text(),
            "Let me analyze this\n\nHere is the code:\n\nDone."
        );
    }

    #[test]
    fn deserialize_tool_result_block() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1"},
                {"type": "text", "text": "Thanks"}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.extract_text(), "Thanks");
    }

    #[test]
    fn extract_text_from_empty_blocks() {
        let content = Content::Blocks(vec![]);
        assert!(content.extract_text().is_empty());
    }

    #[test]
    fn deserialize_full_transcript_entry() {
        let json = r#"{
            "type": "user",
            "uuid": "abc-123",
            "sessionId": "sess-1",
            "timestamp": "2026-02-17T00:00:00Z",
            "message": {"role": "user", "content": "Hello"}
        }"#;
        let entry: TranscriptEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.entry_type.as_deref(), Some("user"));
        assert_eq!(entry.session_id.as_deref(), Some("sess-1"));
        assert!(entry.message.is_some());
    }

    #[test]
    fn deserialize_thinking_block() {
        let json = r#"{
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "I chose X because Y"}]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.extract_text(), "I chose X because Y");
    }

    #[test]
    fn deserialize_thinking_block_none() {
        let json = r#"{
            "role": "assistant",
            "content": [{"type": "thinking"}]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(msg.content.extract_text().is_empty());
    }

    #[test]
    fn deserialize_thinking_block_empty() {
        let json = r#"{
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": ""}]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(msg.content.extract_text().is_empty());
    }

    #[test]
    fn unknown_block_type_skipped() {
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "x"},
                {"type": "text", "text": "result"}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.extract_text(), "result");
        assert!(msg.content.has_unknown_blocks());
    }

    #[test]
    fn thinking_and_text_interleaved() {
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "First thought"},
                {"type": "text", "text": "First response"},
                {"type": "thinking", "thinking": "Second thought"},
                {"type": "text", "text": "Second response"}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(
            msg.content.extract_text(),
            "First thought\n\nFirst response\n\nSecond thought\n\nSecond response"
        );
    }

    #[test]
    fn only_thinking_block_produces_text() {
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Deep reasoning here"}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.extract_text(), "Deep reasoning here");
        assert!(!msg.content.has_unknown_blocks());
    }
}
