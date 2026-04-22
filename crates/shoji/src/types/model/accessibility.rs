use serde::{Deserialize, Serialize};

use crate::types::model::{ModelReference, Repository};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelAccessibility {
    Local {
        reference: ModelReference,
    },
    Remote {
        repository: Option<Repository>,
    },
}
