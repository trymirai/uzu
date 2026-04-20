use shoji::{
    traits::{
        State,
        backend::chat_token::{Backend, Instance},
    },
    types::encoding::Message,
};

use super::Error;

pub struct Session {
    instance: Box<dyn Instance>,
    state: Box<dyn State>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        reference: String,
    ) -> Result<Self, Error> {
        let instance = backend.instance(reference, ()).await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        let state = instance.state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            instance: instance,
            state: state,
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
        _input: Vec<Message>,
    ) -> Result<(), Error> {
        todo!("Implement chat session via token stream")
    }
}
