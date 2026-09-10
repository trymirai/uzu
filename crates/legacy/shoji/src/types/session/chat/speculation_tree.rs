use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeculationTree {
    Argmax {},
    Weaver {
        rounds: u32,
        expand_per_round: u32,
        expand_width: u32,
    },
}
