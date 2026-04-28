mod python;
mod rust;
mod swift;
mod swift_extensions;
mod typescript;

use anyhow::{Result, anyhow};
use colored::Colorize;
pub use python::PythonLanguageBackend;
pub use rust::RustLanguageBackend;
pub use swift::SwiftLanguageBackend;
pub use swift_extensions::generate_swift_extensions;
pub use typescript::TypeScriptLanguageBackend;

use crate::{
    configs::PlatformsConfig,
    types::{Capability, Configuration, Language},
};

#[derive(Clone)]
pub struct LanguageBackendTarget {
    pub name: String,
    pub features: Vec<String>,
}

impl LanguageBackendTarget {
    pub fn new(
        name: String,
        features: Vec<String>,
    ) -> Self {
        Self {
            name,
            features,
        }
    }
}

pub trait LanguageBackend {
    fn config(&self) -> PlatformsConfig;

    fn language(&self) -> Language;

    fn expects_prebuild_for_run(&self) -> bool {
        return true;
    }

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
        configuration: Configuration,
        targets: Vec<String>,
        capabilities: Vec<Capability>,
    ) -> Result<()> {
        let targets = self.resolve_targets("build".to_string(), configuration, targets, capabilities)?;
        self.build_targets(configuration, targets)?;
        Ok(())
    }

    fn build_targets(
        &self,
        _configuration: Configuration,
        _targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        Ok(())
    }

    fn test(
        &self,
        configuration: Configuration,
        target: String,
        capabilities: Vec<Capability>,
    ) -> Result<()> {
        let targets = self.resolve_targets("test".to_string(), configuration, vec![target], capabilities)?;
        if targets.len() != 1 {
            return Err(anyhow!("Expected 1 target for test, got {}", targets.len()));
        }
        let target = targets.first().cloned().ok_or(anyhow!("Target not found"))?;
        self.test_target(configuration, target)?;
        Ok(())
    }

    fn test_target(
        &self,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        Ok(())
    }

    fn example(
        &self,
        name: &str,
        configuration: Configuration,
        target: String,
        capabilities: Vec<Capability>,
    ) -> Result<()> {
        let targets = self.resolve_targets("example".to_string(), configuration, vec![target], capabilities)?;
        if targets.len() != 1 {
            return Err(anyhow!("Expected 1 target for example, got {}", targets.len()));
        }
        let target = targets.first().cloned().ok_or(anyhow!("Target not found"))?;
        self.example_target(name, configuration, target)?;
        Ok(())
    }

    fn example_target(
        &self,
        _name: &str,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        Ok(())
    }

    fn release(
        &self,
        _version: &str,
    ) -> Result<()> {
        Ok(())
    }

    fn resolve_targets(
        &self,
        name: String,
        configuration: Configuration,
        targets: Vec<String>,
        capabilities: Vec<Capability>,
    ) -> Result<Vec<LanguageBackendTarget>> {
        let separator = "--------------------------------------------------".green();
        let language = self.language();
        let bindings = self.config().bindings_for_language(language)?;
        let resolved_targets = self.config().targets_for_language(language, targets.clone())?;
        if resolved_targets.is_empty() {
            return Err(anyhow!("None of the provided targets are supported for language: {language:?}"));
        }
        println!("{separator}");
        println!(
            "Running {} of {} ({}) for targets: {} (resolved from: {})",
            name.green(),
            format!("{:?}", language).green(),
            format!("{:?}", configuration).green(),
            format!("{:?}", resolved_targets).green(),
            format!("{:?}", targets).yellow()
        );
        println!("{separator}");

        let targets = resolved_targets
            .iter()
            .map(|target| {
                println!("Resolving target: {}", target.green());

                let backend = self.config().backend_for_target(target.clone())?;
                let resolved_capabilities =
                    self.config().capabilities_for_target(target.clone(), capabilities.clone())?;
                let features = vec![
                    vec![backend.feature()],
                    resolved_capabilities.iter().map(|capability| capability.feature()).collect::<Vec<_>>(),
                    bindings.iter().map(|binding| binding.feature()).collect::<Vec<_>>(),
                ]
                .iter()
                .flatten()
                .map(|feature| feature.clone())
                .collect::<Vec<_>>();

                println!("Backend: {}", format!("{:?}", backend).green());
                println!(
                    "Capabilities: {} (resolved from: {})",
                    format!("{:?}", resolved_capabilities).green(),
                    format!("{:?}", capabilities).yellow()
                );
                println!("Features: {}", format!("{:?}", features).green());
                println!("{separator}");

                Ok(LanguageBackendTarget::new(target.clone(), features.clone()))
            })
            .collect::<Vec<Result<LanguageBackendTarget>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        Ok(targets)
    }
}
