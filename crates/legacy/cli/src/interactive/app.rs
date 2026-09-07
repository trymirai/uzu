use std::io::IsTerminal;

use iocraft::prelude::*;
use shoji::types::basic::ReasoningEffort;
use uzu::{
    engine::{Engine, EngineConfig, EngineError},
    settings::SettingsError,
};

use crate::interactive::{components::Application, model::ModelResolutionError};

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
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Result<(), CliError> {
        if !std::io::stdout().is_terminal() {
            return Err(CliError::RenderingError {
                message: "stdout is not a terminal".to_string(),
            });
        }

        let settings = self.engine.settings().await.ok();

        let mut application = element! {
            Application(
                engine: Some(self.engine.clone()),
                settings,
                model,
                reasoning_effort,
            )
        };

        #[cfg(all(target_os = "macos", feature = "hardware-control"))]
        let result = {
            use tokio::signal::unix::{SignalKind, signal};

            use super::hardware::HardwareSessionGuard;

            let shutdown_signal = |kind| {
                signal(kind).map_err(|error| CliError::RenderingError {
                    message: format!("Unable to register shutdown signal: {error}"),
                })
            };
            let mut interrupt = shutdown_signal(SignalKind::interrupt())?;
            let mut terminate = shutdown_signal(SignalKind::terminate())?;
            let mut hangup = shutdown_signal(SignalKind::hangup())?;
            let mut controls = HardwareSessionGuard::new();
            application.props.hardware = Some(controls.session());

            let result = tokio::select! {
                result = application.render_loop() => result,
                _ = interrupt.recv() => Ok(()),
                _ = terminate.recv() => Ok(()),
                _ = hangup.recv() => Ok(()),
            };
            if let Err(error) = controls.restore() {
                return Err(CliError::RenderingError {
                    message: match result {
                        Ok(()) => format!("Hardware restoration failed: {error}"),
                        Err(render_error) => format!("{render_error}; hardware restoration failed: {error}"),
                    },
                });
            }
            result
        };
        #[cfg(not(all(target_os = "macos", feature = "hardware-control")))]
        let result = application.render_loop().await;

        result.map_err(|error| CliError::RenderingError {
            message: error.to_string(),
        })
    }
}
