use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Metal,
    Cpu,
}

impl Backend {
    pub fn feature(self) -> String {
        match self {
            Backend::Metal => "backend-metal".to_string(),
            Backend::Cpu => "backend-cpu".to_string(),
        }
    }
}
