use openai_harmony::chat::Role as ExternalRole;
use shoji::types::session::chat::Role;

use crate::chat::harmony::bridging::{Error, FromHarmony, ToHarmony};

impl ToHarmony for Role {
    type Output = ExternalRole;

    fn to_harmony(self) -> Result<Self::Output, Error> {
        match self {
            Role::User {} => Ok(ExternalRole::User),
            Role::Assistant {} => Ok(ExternalRole::Assistant),
            Role::System {} => Ok(ExternalRole::System),
            Role::Developer {} => Ok(ExternalRole::Developer),
            Role::Tool {} => Ok(ExternalRole::Tool),
            Role::Custom {
                ..
            } => Err(Error::UnsupportedRole {
                role: self.clone(),
            }),
        }
    }
}

impl FromHarmony for Role {
    type Input = ExternalRole;

    fn from_harmony(input: Self::Input) -> Self {
        match input {
            ExternalRole::User => Role::User {},
            ExternalRole::Assistant => Role::Assistant {},
            ExternalRole::System => Role::System {},
            ExternalRole::Developer => Role::Developer {},
            ExternalRole::Tool => Role::Tool {},
        }
    }
}
