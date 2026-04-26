mod python;
mod rust;
mod swift;
mod typescript;

use anyhow::{Result, anyhow};
use colored::Colorize;
pub use python::PythonLanguageBackend;
pub use rust::RustLanguageBackend;
pub use swift::SwiftLanguageBackend;
pub use typescript::TypeScriptLanguageBackend;

use crate::{
    configs::PlatformsConfig,
    types::{Capability, Configuration, Language},
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
        targets: Vec<String>,
        capabilities: Vec<Capability>,
    ) -> Result<()> {
        let separator = "--------------------------------------------------".green();
        let language = self.language();
        let resolved_targets = self.config().targets_for_language(language, targets.clone())?;
        if resolved_targets.is_empty() {
            return Err(anyhow!("None of the provided targets are supported for language: {language:?}"));
        }
        println!("{separator}");
        println!(
            "Building for targets: {} (resolved from: {})",
            format!("{:?}", resolved_targets).green(),
            format!("{:?}", targets).yellow()
        );
        println!("{separator}");

        for target in resolved_targets {
            println!("Building for target: {},", target.green());

            let backend = self.config().backend_for_target(target.clone())?;
            let resolved_capabilities = self.config().capabilities_for_target(target.clone(), capabilities.clone())?;
            println!("Backend: {}", format!("{:?}", backend).green());
            println!(
                "Capabilities: {} (resolved from: {})",
                format!("{:?}", resolved_capabilities).green(),
                format!("{:?}", capabilities).yellow()
            );
            println!("{separator}");
        }
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
