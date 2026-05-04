mod components;
mod flows;
mod helpers;
mod sessions;

use std::io::IsTerminal;

use anyhow::Result;
use components::{Application, Theme};
use iocraft::prelude::*;

use crate::engine::{Engine, EngineConfig, EngineError};

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
    pub async fn create(config: EngineConfig) -> Result<Self, EngineError> {
        let engine = Engine::new(config).await?;
        Ok(Self::new(engine))
    }

    #[bindings::export(Method)]
    pub async fn run(&self) -> Result<()> {
        if !std::io::stdout().is_terminal() {
            return Err(anyhow::anyhow!("stdout is not a terminal"));
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
        .await?;

        Ok(())
    }
}
