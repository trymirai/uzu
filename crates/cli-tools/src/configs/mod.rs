mod language;
mod paths;
mod platforms;
mod target;
mod tool;

pub use language::LanguageConfig;
pub use paths::Paths;
pub use platforms::{HOST_TARGET, PlatformsConfig};
pub use target::TargetConfig;
pub use tool::{ToolConfig, ToolProvider};
