use serde::{Deserialize, Serialize};

use crate::types::session::classification::ClassificationRole;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClassificationMessage {
    pub role: ClassificationRole,
    pub content: String,
}

impl ClassificationMessage {
    pub fn new(
        role: ClassificationRole,
        content: String,
    ) -> Self {
        Self {
            role,
            content,
        }
    }
}

#[bindings::export(Implementation)]
impl ClassificationMessage {
    #[bindings::export(Method(Factory))]
    pub fn user(content: String) -> Self {
        Self::new(ClassificationRole::User, content)
    }

    #[bindings::export(Method(Factory))]
    pub fn assistant(content: String) -> Self {
        Self::new(ClassificationRole::Assistant, content)
    }
}
