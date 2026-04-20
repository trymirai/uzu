mod error;
pub mod message;
pub mod token;

pub use error::Error;
use shoji::{
    traits::Backend,
    types::{
        encoding::Message,
        model::{Model, Specialization},
    },
};

pub enum Session {
    Token(token::Session),
    Message(message::Session),
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, Error> {
        if !model.specializations.contains(&Specialization::Chat) {
            return Err(Error::UnsupportedModel);
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        if let Some(token_backend) = backend.as_chat_via_token_capable() {
            let session = token::Session::new(token_backend, reference).await?;
            return Ok(Self::Token(session));
        }

        if let Some(message_backend) = backend.as_chat_via_message_capable() {
            let session = message::Session::new(message_backend, reference).await?;
            return Ok(Self::Message(session));
        }

        Err(Error::UnsupportedModel)
    }

    pub async fn reset(&mut self) -> Result<(), Error> {
        match self {
            Session::Token(session) => session.reset().await,
            Session::Message(session) => session.reset().await,
        }
    }

    pub async fn stream(
        &mut self,
        input: Vec<Message>,
    ) -> Result<(), Error> {
        match self {
            Session::Token(session) => session.stream(input).await,
            Session::Message(session) => session.stream(input).await,
        }
    }
}
