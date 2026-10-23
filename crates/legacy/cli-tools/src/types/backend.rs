use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Vulkan,
    Cpu,
}

impl Backend {
    pub fn name(self) -> String {
        match self {
            Backend::Metal => "metal".to_string(),
            Backend::Vulkan => "vulkan".to_string(),
            Backend::Cpu => "cpu".to_string(),
        }
    }

    pub fn feature(self) -> String {
        match self {
            Backend::Metal => "backend-metal".to_string(),
            // The uzu crate enables Vulkan by target, not by feature, and the engine's Vulkan includes the CPU backend.
            Backend::Vulkan | Backend::Cpu => "backend-cpu".to_string(),
        }
    }
}
