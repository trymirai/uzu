use super::{chat_turn::ChatTurn, role::Role};

pub(super) fn conversation_for_request(messages: &[ChatTurn]) -> Vec<(Role, String)> {
    let upto = messages.len().saturating_sub(1);
    messages[..upto].iter().filter(|m| !m.cur().error).map(|m| (m.role, m.cur().text.clone())).collect()
}
