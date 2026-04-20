use futures::StreamExt;
use shoji::{
    traits::{
        State,
        backend::chat_message::{Backend, Instance},
    },
    types::{
        encoding::Message,
        session::chat::{Config, StreamConfig},
    },
};
use tokio_util::sync::CancellationToken;

use super::Error;

pub struct Session {
    instance: Box<dyn Instance>,
    state: Box<dyn State>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: Config,
        reference: String,
    ) -> Result<Self, Error> {
        let instance = backend.instance(reference, config).await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        let state = instance.state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            instance,
            state,
        })
    }

    pub async fn reset(&mut self) -> Result<(), Error> {
        self.state = self.instance.state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        Ok(())
    }

    pub async fn stream(
        &mut self,
        input: Vec<Message>,
        config: StreamConfig,
    ) -> Result<(), Error> {
        let mut stream = self.instance.stream(&input, self.state.as_mut(), config, CancellationToken::new());
        while let Some(event) = stream.next().await {
            match event {
                Ok(output) => println!("{output:?}"),
                Err(error) => eprintln!("stream error: {error}"),
            }
        }
        Ok(())
    }
}
