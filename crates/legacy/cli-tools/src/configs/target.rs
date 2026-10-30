use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TargetConfig {
    pub features: Vec<String>,
    pub aliases: Vec<String>,
    #[serde(default)]
    pub required_envs: Vec<String>,
}
