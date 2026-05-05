mod components;
mod flows;
mod helpers;
mod sessions;

use std::io::IsTerminal;

use components::{Application, Theme};
use iocraft::prelude::*;

use crate::{
    engine::{Engine, EngineConfig, EngineError},
    settings::SettingsError,
};

#[bindings::export(Error)]
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

#[bindings::export(Class)]
#[derive(Clone)]
pub struct CliApplication {
    engine: Engine,
}

impl CliApplication {
    pub fn new(engine: Engine) -> Self {
        Self {
            engine,
        }
    }
}

#[bindings::export(Implementation)]
impl CliApplication {
    #[bindings::export(Method(Factory))]
    pub async fn create(config: EngineConfig) -> Result<Self, CliError> {
        let engine = Engine::new(config).await?;
        Ok(Self::new(engine))
    }

    #[bindings::export(Method)]
    pub async fn run(&self) -> Result<(), CliError> {
        if !std::io::stdout().is_terminal() {
            return Err(CliError::RenderingError {
                message: "stdout is not a terminal".to_string(),
            });
        }

        let settings = self.engine.settings().await.ok();
        let theme = match &settings {
            Some(settings) => Theme::load(settings)?.unwrap_or_default(),
            None => Theme::default(),
        };

        element! {
            Application(engine: Some(self.engine.clone()), settings: settings, theme: Some(theme))
        }
        .render_loop()
        .await
        .map_err(|error| CliError::RenderingError {
            message: error.to_string(),
        })?;

        Ok(())
    }
}
