use std::fmt::Write as _;

use text_splitter::{ChunkConfig, MarkdownSplitter};
use tokenizers::Tokenizer;

use crate::config::{CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS};
use crate::transcript::parser::ParsedMessage;

/// A turn groups consecutive messages for semantic coherence:
/// Turn[n] = User[n] + Assistant[n] + User[n+1]
#[derive(Debug)]
pub struct Turn {
    /// JSONL line index of the first message in this turn (User[n]).
    pub line_index: usize,
    /// Whether this turn is provisional (missing User[n+1]).
    pub provisional: bool,
    /// Combined text content of the turn.
    pub text: String,
}

/// Group parsed messages into turns.
/// A turn = [User[n] + Assistant[n] + User[n+1]].
/// The last turn is provisional if there's no following User message.
pub fn group_into_turns(messages: &[ParsedMessage]) -> Vec<Turn> {
    // Collect user-assistant pairs
    let mut pairs: Vec<(usize, &ParsedMessage, &ParsedMessage)> = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        if messages[i].role == "user"
            && i + 1 < messages.len()
            && messages[i + 1].role == "assistant"
        {
            pairs.push((i, &messages[i], &messages[i + 1]));
            i += 2;
        } else {
            i += 1;
        }
    }

    if pairs.is_empty() {
        return Vec::new();
    }

    let mut turns = Vec::new();

    for (idx, (_, user, assistant)) in pairs.iter().enumerate() {
        let mut text = format!("[User] {}\n\n[Assistant] {}", user.text, assistant.text);
        let mut provisional = true;

        // If there's a next pair, its user message is our forward context
        if idx + 1 < pairs.len() {
            let (_, next_user, _) = &pairs[idx + 1];
            write!(&mut text, "\n\n[User] {}", next_user.text).unwrap();
            provisional = false;
        }

        turns.push(Turn {
            line_index: user.line_index,
            provisional,
            text,
        });
    }

    turns
}

/// A chunk ready for embedding.
#[derive(Debug)]
pub struct Chunk {
    /// JSONL line index of the turn this chunk belongs to.
    pub line_index: usize,
    /// Sequential chunk index within the turn (0-based).
    pub chunk_index: usize,
    /// The text content of this chunk.
    pub text: String,
}

/// Split a turn's text into sub-chunks with overlap.
pub fn chunk_turn(turn: &Turn, tokenizer: &Tokenizer) -> Vec<Chunk> {
    let splitter =
        MarkdownSplitter::new(ChunkConfig::new(CHUNK_TARGET_TOKENS).with_sizer(tokenizer));
    let raw_chunks: Vec<&str> = splitter.chunks(&turn.text).collect();

    if raw_chunks.is_empty() {
        return Vec::new();
    }

    if raw_chunks.len() == 1 {
        return vec![Chunk {
            line_index: turn.line_index,
            chunk_index: 0,
            text: raw_chunks[0].to_string(),
        }];
    }

    // Apply overlap: prepend last N tokens from previous chunk
    let mut chunks = vec![Chunk {
        line_index: turn.line_index,
        chunk_index: 0,
        text: raw_chunks[0].to_string(),
    }];

    for (i, &chunk_text) in raw_chunks.iter().enumerate().skip(1) {
        let overlap_prefix =
            extract_tail_tokens(raw_chunks[i - 1], tokenizer, CHUNK_OVERLAP_TOKENS);
        let text = if overlap_prefix.is_empty() {
            chunk_text.to_string()
        } else {
            format!("{overlap_prefix}\n\n{chunk_text}")
        };

        chunks.push(Chunk {
            line_index: turn.line_index,
            chunk_index: i,
            text,
        });
    }

    chunks
}

/// Extract the last `n_tokens` worth of text from the end of `text`.
fn extract_tail_tokens(text: &str, tokenizer: &Tokenizer, n_tokens: usize) -> String {
    let Ok(encoding) = tokenizer.encode(text, false) else {
        return String::new();
    };

    let tokens = encoding.get_ids();
    if tokens.is_empty() {
        return String::new();
    }

    let start = tokens.len().saturating_sub(n_tokens);
    let tail_ids = &tokens[start..];

    tokenizer.decode(tail_ids, true).unwrap_or_default()
}

/// Load the tokenizer for BGE-small-en-v1.5 from bundled files.
pub fn load_tokenizer() -> anyhow::Result<Tokenizer> {
    let tokenizer_bytes = include_bytes!("../../../../models/bge-small-en-v1.5/tokenizer.json");
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages(roles_and_texts: &[(&str, &str)]) -> Vec<ParsedMessage> {
        roles_and_texts
            .iter()
            .enumerate()
            .map(|(i, (role, text))| ParsedMessage {
                line_index: i,
                role: (*role).to_string(),
                text: (*text).to_string(),
            })
            .collect()
    }

    #[test]
    fn single_pair_is_provisional() {
        let msgs = make_messages(&[("user", "Hello"), ("assistant", "Hi there")]);
        let turns = group_into_turns(&msgs);
        assert_eq!(turns.len(), 1);
        assert!(turns[0].provisional);
        assert!(turns[0].text.contains("[User] Hello"));
        assert!(turns[0].text.contains("[Assistant] Hi there"));
    }

    #[test]
    fn two_pairs_first_is_complete() {
        let msgs = make_messages(&[
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2"),
        ]);
        let turns = group_into_turns(&msgs);
        assert_eq!(turns.len(), 2);
        assert!(!turns[0].provisional);
        assert!(turns[0].text.contains("[User] Q2")); // forward context
        assert!(turns[1].provisional);
    }

    #[test]
    fn inter_turn_overlap_shares_user_message() {
        let msgs = make_messages(&[
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2"),
            ("user", "Q3"),
            ("assistant", "A3"),
        ]);
        let turns = group_into_turns(&msgs);
        assert_eq!(turns.len(), 3);

        // Q2 appears in both Turn 0 (forward context) and Turn 1 (start)
        assert!(turns[0].text.contains("[User] Q2"));
        assert!(turns[1].text.starts_with("[User] Q2"));
    }

    #[test]
    fn empty_messages_produces_no_turns() {
        let turns = group_into_turns(&[]);
        assert!(turns.is_empty());
    }

    #[test]
    fn short_turn_single_chunk() {
        let tokenizer = load_tokenizer().unwrap();
        let turn = Turn {
            line_index: 0,
            provisional: false,
            text: "Short text that fits in one chunk.".to_string(),
        };
        let chunks = chunk_turn(&turn, &tokenizer);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn long_turn_multiple_chunks() {
        let tokenizer = load_tokenizer().unwrap();
        // Generate text longer than 256 tokens
        let long_text = "This is a sentence that adds some tokens. ".repeat(100);
        let turn = Turn {
            line_index: 0,
            provisional: false,
            text: long_text,
        };
        let chunks = chunk_turn(&turn, &tokenizer);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );

        // Verify sequential chunk indices
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
        }
    }

    #[test]
    fn sub_chunk_overlap_present() {
        let tokenizer = load_tokenizer().unwrap();
        let long_text = "This is a sentence that adds some tokens. ".repeat(100);
        let turn = Turn {
            line_index: 0,
            provisional: false,
            text: long_text,
        };
        let chunks = chunk_turn(&turn, &tokenizer);
        if chunks.len() > 1 {
            // Second chunk should contain overlap prefix
            // The overlap prefix comes from the tail of the first chunk
            assert!(
                chunks[1].text.len() > chunks[0].text.len() / 4,
                "Second chunk seems too short — overlap may be missing"
            );
        }
    }

    #[test]
    fn extract_tail_tokens_basic() {
        let tokenizer = load_tokenizer().unwrap();
        let text = "Hello world this is a test of the tokenizer";
        let tail = extract_tail_tokens(text, &tokenizer, 3);
        assert!(!tail.is_empty());
        // The tail should be the last few words
        assert!(text.ends_with(&tail) || tail.len() < text.len());
    }

    #[test]
    fn extract_tail_tokens_empty_text() {
        let tokenizer = load_tokenizer().unwrap();
        let tail = extract_tail_tokens("", &tokenizer, 10);
        assert!(tail.is_empty());
    }

    #[test]
    fn turn_line_indices_preserved() {
        let msgs = vec![
            ParsedMessage {
                line_index: 5,
                role: "user".to_string(),
                text: "Q1".to_string(),
            },
            ParsedMessage {
                line_index: 6,
                role: "assistant".to_string(),
                text: "A1".to_string(),
            },
            ParsedMessage {
                line_index: 10,
                role: "user".to_string(),
                text: "Q2".to_string(),
            },
            ParsedMessage {
                line_index: 11,
                role: "assistant".to_string(),
                text: "A2".to_string(),
            },
        ];
        let turns = group_into_turns(&msgs);
        assert_eq!(turns[0].line_index, 5);
        assert_eq!(turns[1].line_index, 10);
    }
}
