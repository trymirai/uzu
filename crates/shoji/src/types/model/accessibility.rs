use serde::{Deserialize, Serialize};

use crate::types::model::{Reference, Repository};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Accessibility {
    Local {
        reference: Reference,
    },
    Remote {
        repository: Option<Repository>,
    },
}
