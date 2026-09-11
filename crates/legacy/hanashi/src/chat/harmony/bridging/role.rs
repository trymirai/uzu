use openai_harmony::chat::Role as ExternalRole;
use shoji::types::session::chat::ChatRole;

use crate::chat::harmony::bridging::{Error, FromHarmony, ToHarmony};

impl ToHarmony for ChatRole {
    type Output = ExternalRole;

    fn to_harmony(self) -> Result<Self::Output, Error> {
        match self {
            ChatRole::User {} => Ok(ExternalRole::User),
            ChatRole::Assistant {} => Ok(ExternalRole::Assistant),
            ChatRole::System {} => Ok(ExternalRole::System),
            ChatRole::Developer {} => Ok(ExternalRole::Developer),
            ChatRole::Tool {} => Ok(ExternalRole::Tool),
            ChatRole::Custom {
                ..
            } => Err(Error::UnsupportedRole {
                role: self.clone(),
            }),
        }
    }
}

impl FromHarmony for ChatRole {
    type Input = ExternalRole;

    fn from_harmony(input: Self::Input) -> Self {
        match input {
            ExternalRole::User => ChatRole::User {},
            ExternalRole::Assistant => ChatRole::Assistant {},
            ExternalRole::System => ChatRole::System {},
            ExternalRole::Developer => ChatRole::Developer {},
            ExternalRole::Tool => ChatRole::Tool {},
        }
    }
}
