use std::io::IsTerminal;

use iocraft::prelude::*;
use uzu::{
    engine::{Engine, EngineConfig, EngineError},
    settings::SettingsError,
};

use crate::interactive::{
    components::{AppSettings, Application, Preferences, Theme},
    model::{ModelResolutionError, resolve_model_id},
};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CliError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    ModelResolution(#[from] ModelResolutionError),
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
    ) -> Result<(), CliError> {
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
        let preferences = match &settings {
            Some(settings) => Preferences::load(settings)?,
            None => Preferences::default(),
        };
        let app_settings = match &settings {
            Some(settings) => AppSettings::load(settings)?,
            None => AppSettings::default(),
        };

        let requested_model = model.or_else(|| app_settings.selected_model_id.clone());
        let selected_model = match requested_model {
            Some(model) => resolve_model_id(&self.engine, model).await?,
            None => None,
        };

        element! {
            Application(
                engine: Some(self.engine.clone()),
                settings: settings,
                theme: Some(theme),
                preferences: Some(preferences),
                app_settings: Some(app_settings),
                model: selected_model,
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
