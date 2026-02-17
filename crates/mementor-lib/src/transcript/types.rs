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
}

impl Content {
    /// Extract all text content, skipping `tool_use` and `tool_result` blocks.
    pub fn extract_text(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Blocks(blocks) => {
                let texts: Vec<&str> = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => None,
                    })
                    .collect();
                texts.join("\n\n")
            }
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
                {"type": "text", "text": "Here is the code:"},
                {"type": "tool_use", "id": "t1", "name": "write"},
                {"type": "text", "text": "Done."}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.extract_text(), "Here is the code:\n\nDone.");
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
}
