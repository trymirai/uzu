use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shoji::types::session::chat::Role;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub initial: Vec<Role>,
    pub transitions: IndexMap<Role, Vec<Role>>,
}

impl Config {
    pub fn is_role_avoidable(
        &self,
        role: &Role,
    ) -> bool {
        let other_initials = self.initial.iter().any(|initial| initial != role);
        let all_transitions_have_alternatives = self
            .transitions
            .values()
            .all(|targets| !targets.contains(role) || targets.iter().any(|target| target != role));
        other_initials && all_transitions_have_alternatives
    }
}
