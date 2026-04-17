use serde::{Deserialize, Serialize};

use crate::registry::types::{Reference, Repository};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Accessibility {
    Local {
        provider_identifier: String,
        reference: Reference,
    },
    Remote {
        provider_identifier: String,
        source_repository: Option<Repository>,
    },
}
