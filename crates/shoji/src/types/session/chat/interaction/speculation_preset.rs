use serde::{Deserialize, Serialize};

use crate::types::session::chat::Feature;

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeculationPreset {
    GeneralChat {},
    Summarization {},
    Classification {
        feature: Feature,
    },
}
