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
}

impl ConfigResolvableValue<MessageProcessorConfig, HashMap<String, String>>
    for Message
{
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
        HashMap::from([
            (String::from("role"), role),
            (String::from("content"), content),
        ])
    }
}
