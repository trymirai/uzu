use serde::{Deserialize, Serialize};

use crate::types::{Backend, Capability};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TargetConfig {
    pub backend: Backend,
    pub aliases: Vec<String>,
    pub capabilities: Vec<Capability>,
}
