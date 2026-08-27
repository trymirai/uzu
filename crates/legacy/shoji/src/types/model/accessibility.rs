use serde::{Deserialize, Serialize};

use crate::types::{basic::Repository, model::ModelSource};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelAccessibility {
    OnDevice {
        source: ModelSource,
    },
    Remote {
        repository: Option<Repository>,
    },
}
