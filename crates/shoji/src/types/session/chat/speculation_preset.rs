use serde::{Deserialize, Serialize};

use crate::types::basic::Feature;

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatSpeculationPreset {
    GeneralChat {},
    Summarization {},
    Classification {
        feature: Feature,
    },
}
