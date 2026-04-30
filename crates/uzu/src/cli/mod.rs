mod components;

use std::io::IsTerminal;

use anyhow::Result;
use components::Application;
use dotenvy::dotenv;
use iocraft::prelude::*;

use crate::engine::Engine;

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

impl CliApplication {
    pub async fn run(&self) -> Result<()> {
        if !std::io::stdout().is_terminal() {
            return Err(anyhow::anyhow!("stdout is not a terminal"));
        }

        dotenv().ok();
        element! {
            Application(engine: Some(self.engine.clone()))
        }
        .render_loop()
        .await?;

        Ok(())
    }
}
