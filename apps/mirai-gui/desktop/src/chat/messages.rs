use uzu::types::{
    basic::ReasoningEffort,
    session::chat::{ChatMessage, ChatRole},
};

use super::payloads::{MsgIn, Parsed};

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

pub(super) fn build_non_system_message(
    message: &MsgIn,
    include_reasoning: bool,
) -> ChatMessage {
    let content = message.content.clone();
    if matches!(message.role, ChatRole::User {}) {
        return ChatMessage::user().with_text(content);
    }

    let mut assistant = ChatMessage::assistant();
    if include_reasoning && let Some(reasoning) = message.reasoning_content.as_deref().filter(|value| !value.is_empty())
    {
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
            _ => build_non_system_message(m, false),
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

// uzu rejects another system message in an accumulated session.
pub(super) fn build_tail_messages(tail: &[MsgIn]) -> Vec<ChatMessage> {
    tail.iter().filter(|m| !matches!(m.role, ChatRole::System {})).map(|m| build_non_system_message(m, true)).collect()
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
    if content != text || reasoning_content != reasoning {
        return None;
    }
    Some(MsgIn {
        role: ChatRole::Assistant {},
        content,
        reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
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
    fn full_replay_omits_assistant_reasoning_blocks() {
        let messages = build_messages(&[test_message(ChatRole::Assistant {}, "answer", Some("reasoning"))], None);
        assert_eq!(messages[0].text().as_deref(), Some("answer"));
        assert_eq!(messages[0].reasoning(), None);
    }

    #[test]
    fn unsafe_generated_history_is_not_reused() {
        assert!(assistant_history_message("broken \u{FFFD}", "", false).is_none());
        assert!(assistant_history_message("emoji \u{2764}\u{FE0F}", "", false).is_some());
        assert!(assistant_history_message("answer", " reasoning ", false).is_none());
        assert!(assistant_history_message("answer", "reasoning", true).is_none());
    }

    #[test]
    fn exact_generated_history_keeps_reasoning() {
        let message = assistant_history_message("answer", "reasoning", false).expect("history message");
        assert_eq!(message.content, "answer");
        assert_eq!(message.reasoning_content.as_deref(), Some("reasoning"));
    }
}
