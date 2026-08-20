use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shoji::types::session::chat::ChatRole;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub initial: Vec<ChatRole>,
    pub transitions: IndexMap<ChatRole, Vec<ChatRole>>,
}

impl Config {
    pub fn is_role_avoidable(
        &self,
        role: &ChatRole,
    ) -> bool {
        let other_initials = self.initial.iter().any(|initial| initial != role);
        let all_transitions_have_alternatives = self
            .transitions
            .values()
            .all(|targets| !targets.contains(role) || targets.iter().any(|target| target != role));
        other_initials && all_transitions_have_alternatives
    }
}
