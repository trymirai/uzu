mod config;
mod context;
mod error;
pub mod hanashi;
pub mod harmony;
mod state;

pub use config::Config;
pub use context::{Context, TokenizerLocation};
pub use error::Error;
use shoji::types::{basic::TokenId, session::chat::ChatMessage};
pub use state::{State, SynchronizationError, SynchronizationResult};

use crate::{
    Encoding as EncodingTrait,
    chat::{hanashi::Encoding as HanashiEncoding, harmony::Encoding as HarmonyEncoding},
};

macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Encoding::Hanashi(inner) => inner.$method($($arg),*).map_err(Into::into),
            Encoding::Harmony(inner) => inner.$method($($arg),*).map_err(Into::into),
        }
    };
    (infallible $self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Encoding::Hanashi(inner) => inner.$method($($arg),*),
            Encoding::Harmony(inner) => inner.$method($($arg),*),
        }
    };
}

pub enum Encoding {
    Hanashi(HanashiEncoding),
    Harmony(HarmonyEncoding),
}

impl EncodingTrait for Encoding {
    type Config = Config;
    type Context = Context;
    type Input = Vec<ChatMessage>;
    type Output = Vec<TokenId>;
    type State = State;
    type Error = Error;

    fn new(
        config: Self::Config,
        context: Self::Context,
    ) -> Result<Self, Self::Error> {
        match config {
            Config::Hanashi(config) => Ok(Encoding::Hanashi(HanashiEncoding::new(config, context)?)),
            Config::Harmony(config) => Ok(Encoding::Harmony(HarmonyEncoding::new(config, context)?)),
        }
    }

    fn state(&self) -> &Self::State {
        dispatch!(infallible self, state)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        dispatch!(self, reset)
    }

    fn encode(
        &mut self,
        value: Self::Input,
    ) -> Result<(), Self::Error> {
        dispatch!(self, encode, value)
    }

    fn decode(
        &mut self,
        value: Self::Output,
    ) -> Result<(), Self::Error> {
        dispatch!(self, decode, value)
    }
}
