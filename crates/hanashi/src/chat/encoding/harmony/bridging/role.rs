use openai_harmony::chat::Role as ExternalRole;

use crate::chat::{encoding::harmony::bridging::Error, types::Role};

impl TryFrom<Role> for ExternalRole {
    type Error = Error;

    fn try_from(role: Role) -> Result<ExternalRole, Error> {
        match role {
            Role::User {} => Ok(ExternalRole::User),
            Role::Assistant {} => Ok(ExternalRole::Assistant),
            Role::System {} => Ok(ExternalRole::System),
            Role::Developer {} => Ok(ExternalRole::Developer),
            Role::Tool {} => Ok(ExternalRole::Tool),
            Role::Custom {
                ..
            } => Err(Error::UnsupportedRole {
                role: role.clone(),
            }),
        }
    }
}

impl From<ExternalRole> for Role {
    fn from(role: ExternalRole) -> Role {
        match role {
            ExternalRole::User => Role::User {},
            ExternalRole::Assistant => Role::Assistant {},
            ExternalRole::System => Role::System {},
            ExternalRole::Developer => Role::Developer {},
            ExternalRole::Tool => Role::Tool {},
        }
    }
}
