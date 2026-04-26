mod python;
mod rust;
mod swift;
mod typescript;

use anyhow::Result;
pub use python::PythonLanguageBackend;
pub use rust::RustLanguageBackend;
pub use swift::SwiftLanguageBackend;
pub use typescript::TypeScriptLanguageBackend;

use crate::{
    configs::PlatformsConfig,
    types::{Configuration, Language},
};

pub trait LanguageBackend {
    fn config(&self) -> PlatformsConfig;
    fn language(&self) -> Language;

    fn install(&self) -> Result<()> {
        let language = self.language();
        let tools = self.config().tools_for_language(language)?;
        for tool in tools {
            let command = tool.command();
            command.run()?;
        }
        Ok(())
    }

    fn build(
        &self,
        _configuration: Configuration,
    ) -> Result<()> {
        Ok(())
    }

    fn test(&self) -> Result<()> {
        Ok(())
    }

    fn example(
        &self,
        _name: &str,
    ) -> Result<()> {
        Ok(())
    }
}
