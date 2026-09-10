use serde::{Deserialize, Serialize};

use super::SpeculationShape;

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeculationMode {
    Auto {},
    Off {},
    Shape {
        shape: SpeculationShape,
    },
}

impl Default for SpeculationMode {
    fn default() -> Self {
        SpeculationMode::Auto {}
    }
}
