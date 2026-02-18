use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    config::MessageProcessorConfig,
    session::{parameter::ConfigResolvableValue, types::Role},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(
        role: Role,
        content: String,
        reasoning_content: Option<String>,
    ) -> Self {
        Self {
            role,
            content,
            reasoning_content,
        }
    }

    pub fn system(content: String) -> Self {
        Self::new(Role::System, content, None)
    }

    pub fn user(content: String) -> Self {
        Self::new(Role::User, content, None)
    }

    pub fn assistant(
        content: String,
        reasoning_content: Option<String>,
    ) -> Self {
        Self::new(Role::Assistant, content, reasoning_content)
    }
}

impl ConfigResolvableValue<MessageProcessorConfig, HashMap<String, String>> for Message {
    fn resolve(
        &self,
        config: &MessageProcessorConfig,
    ) -> HashMap<String, String> {
        let role = match self.role {
            Role::System => config.system_role_name.clone(),
            Role::User => config.user_role_name.clone(),
            Role::Assistant => config.assistant_role_name.clone(),
        };
        let content = self.content.clone();
        let mut result = HashMap::from([(String::from("role"), role), (String::from("content"), content)]);
        if let Some(reasoning_content) = self.reasoning_content.clone() {
            result.insert(String::from("reasoning_content"), reasoning_content);
        }
        result
    }
}
