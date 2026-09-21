use std::io::IsTerminal;

use iocraft::prelude::*;
use shoji::types::basic::ReasoningEffort;
use uzu::{
    engine::{Engine, EngineConfig, EngineError},
    settings::SettingsError,
};

use crate::interactive::components::Application;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CliError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Settigs(#[from] SettingsError),
    #[error("Rendering error: {message}")]
    RenderingError {
        message: String,
    },
}

#[derive(Clone)]
pub struct CliApplication {
    engine: Engine,
}

impl CliApplication {
    pub async fn create(config: EngineConfig) -> Result<Self, CliError> {
        let engine = Engine::new(config).await?;
        Ok(Self::new(engine))
    }

    pub fn new(engine: Engine) -> Self {
        Self {
            engine,
        }
    }

    pub async fn run_with_model(
        &self,
        model: Option<String>,
        reasoning_effort: Option<ReasoningEffort>,
        seed: Option<i64>,
        no_tools: bool,
    ) -> Result<(), CliError> {
        if !std::io::stdout().is_terminal() {
            return Err(CliError::RenderingError {
                message: "stdout is not a terminal".to_string(),
            });
        }

        let settings = self.engine.settings().await.ok();

        element! {
            Application(
                engine: Some(self.engine.clone()),
                settings,
                model,
                reasoning_effort,
                seed,
                no_tools,
            )
        }
        .render_loop()
        .await
        .map_err(|error| CliError::RenderingError {
            message: error.to_string(),
        })?;

        Ok(())
    }
}
