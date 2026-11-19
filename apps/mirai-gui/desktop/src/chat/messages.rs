use uzu::types::{
    basic::ReasoningEffort,
    session::chat::{ChatMessage, ChatRole},
};

use super::payloads::{MsgIn, Parsed};

pub(super) fn clip_chars(
    text: &str,
    max: usize,
) -> String {
    match text.char_indices().nth(max) {
        Some(_) => {
            let kept: String = text.chars().take(max.saturating_sub(1)).collect();
            format!("{}…", kept.trim_end())
        },
        None => text.to_string(),
    }
}

pub(super) fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// Engine output only: U+FFFD shows up as "�" when the decoder splits a multi-byte
// character across tokens. Joiners and variation selectors hold composite emoji together.
pub(super) fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|&c| {
            !((c.is_control() && c != '\n' && c != '\r' && c != '\t')
                || c == '\u{FFFD}'
                || c == '\u{FFFC}'
                || c == '\u{FEFF}')
        })
        .collect()
}

fn build_non_system_message(message: &MsgIn) -> ChatMessage {
    let content = message.content.clone();
    if matches!(message.role, ChatRole::User {}) {
        return ChatMessage::user().with_text(content);
    }

    let mut assistant = ChatMessage::assistant();
    if let Some(reasoning) = message.reasoning_content.as_deref().filter(|value| !value.is_empty()) {
        assistant = assistant.with_reasoning(reasoning.to_string());
    }
    assistant.with_text(content)
}

// uzu reads reasoning effort from a system-message block.
pub(super) fn build_messages(
    raw: &[MsgIn],
    reasoning_effort: Option<ReasoningEffort>,
) -> Vec<ChatMessage> {
    let mut messages: Vec<ChatMessage> = raw
        .iter()
        .map(|m| match m.role {
            ChatRole::System {} => ChatMessage::system().with_text(m.content.clone()),
            _ => build_non_system_message(m),
        })
        .collect();

    if let Some(effort) = reasoning_effort {
        if let Some(index) = raw.iter().position(|m| matches!(m.role, ChatRole::System {})) {
            messages[index] = messages[index].with_reasoning_effort(effort);
        } else {
            messages.insert(0, ChatMessage::system().with_reasoning_effort(effort));
        }
    }
    messages
}

pub(super) fn append_system_instruction(
    messages: &mut Vec<ChatMessage>,
    instruction: String,
) {
    if let Some(system) = messages.iter_mut().find(|message| message.role == (ChatRole::System {})) {
        *system = system.with_text(format!("\n\n{instruction}"));
    } else {
        messages.insert(0, ChatMessage::system().with_text(instruction));
    }
}

// uzu rejects another system message in an accumulated session.
pub(super) fn build_tail_messages(tail: &[MsgIn]) -> Vec<ChatMessage> {
    tail.iter().filter(|m| !matches!(m.role, ChatRole::System {})).map(build_non_system_message).collect()
}

pub(super) fn assistant_history_message(
    text: &str,
    reasoning: &str,
    promoted_reasoning: bool,
) -> Option<MsgIn> {
    if promoted_reasoning || text.trim().is_empty() {
        return None;
    }
    let content = sanitize(text);
    let reasoning_content = sanitize(reasoning.trim());
    // Match the final text/parsed fields sent to the client. The resident session
    // keeps the original reasoning bytes and internal tool messages.
    Some(MsgIn {
        role: ChatRole::Assistant {},
        content,
        reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
        openable_urls: Vec::new(),
    })
}

pub(super) fn parsed_from(
    text: &str,
    reasoning: &str,
) -> Option<Parsed> {
    let chain = reasoning.trim();
    let resp = text.trim();
    if chain.is_empty() && resp.is_empty() {
        return None;
    }
    Some(Parsed {
        chain_of_thought: (!chain.is_empty()).then(|| sanitize(chain)),
        response: (!resp.is_empty()).then(|| sanitize(resp)),
    })
}

#[cfg(test)]
mod tests {
    use super::{super::payloads::test_message, *};

    #[test]
    fn sanitize_drops_replacement_and_control_characters_but_keeps_emoji_sequences() {
        assert_eq!(sanitize("a\u{FFFD}b\u{0000}c\u{FEFF}d\n"), "abcd\n");
        assert_eq!(
            sanitize("👩\u{200D}💻 ❤\u{FE0F} 👨\u{200D}👩\u{200D}👧"),
            "👩\u{200D}💻 ❤\u{FE0F} 👨\u{200D}👩\u{200D}👧"
        );
    }

    #[test]
    fn user_text_reaches_the_model_as_typed() {
        let messages = build_messages(&[test_message(ChatRole::User {}, "what does \u{FFFD} mean?", None)], None);
        assert_eq!(messages[0].text().as_deref(), Some("what does \u{FFFD} mean?"));
    }

    #[test]
    fn incremental_tail_keeps_assistant_reasoning_blocks() {
        let messages = build_tail_messages(&[test_message(ChatRole::Assistant {}, "answer", Some("reasoning"))]);
        assert_eq!(messages[0].text().as_deref(), Some("answer"));
        assert_eq!(messages[0].reasoning().as_deref(), Some("reasoning"));
    }

    #[test]
    fn full_replay_keeps_assistant_reasoning_blocks() {
        let messages = build_messages(&[test_message(ChatRole::Assistant {}, "answer", Some("reasoning"))], None);
        assert_eq!(messages[0].text().as_deref(), Some("answer"));
        assert_eq!(messages[0].reasoning().as_deref(), Some("reasoning"));
    }

    #[test]
    fn incomplete_or_promoted_generated_history_is_not_reused() {
        assert!(assistant_history_message("\n ", "reasoning", false).is_none());
        assert!(assistant_history_message("answer", "reasoning", true).is_none());
    }

    #[test]
    fn exact_generated_history_keeps_reasoning() {
        let message = assistant_history_message("answer", "reasoning", false).expect("history message");
        assert_eq!(message.content, "answer");
        assert_eq!(message.reasoning_content.as_deref(), Some("reasoning"));
    }

    #[test]
    fn an_instruction_joins_the_system_message_or_opens_one() {
        let mut with_system = vec![
            ChatMessage::system().with_text("Be helpful.".to_string()),
            ChatMessage::user().with_text("Hi".to_string()),
        ];
        append_system_instruction(&mut with_system, "Be brief.".to_string());
        assert_eq!(with_system.len(), 2);
        assert_eq!(with_system[0].text().unwrap(), "Be helpful.\n\nBe brief.");

        let mut without_system = vec![ChatMessage::user().with_text("Hi".to_string())];
        append_system_instruction(&mut without_system, "Be brief.".to_string());
        assert_eq!(without_system.len(), 2);
        assert_eq!(without_system[0].role, ChatRole::System {});
        assert_eq!(without_system[0].text().unwrap(), "Be brief.");
    }

    #[test]
    fn clipped_text_never_exceeds_the_limit_and_ends_with_an_ellipsis() {
        assert_eq!(clip_chars("naïve café", 20), "naïve café");
        assert_eq!(clip_chars("abc", 3), "abc");
        assert_eq!(clip_chars("één twee drie", 9), "één twee…");
        assert_eq!(clip_chars("één twee drie", 5), "één…");
    }
}
