mod config;
mod context;
mod error;
pub mod hanashi;
pub mod harmony;
mod state;

pub use config::EncodingConfig;
pub use context::TokenizerLocation;
pub use error::Error;
pub use hanashi::renderer::strftime_now;
use shoji::types::{basic::TokenId, session::chat::ChatMessage};
pub use state::{State, SynchronizationError, SynchronizationResult};

use crate::{
    Encoding as EncodingTrait,
    chat::{hanashi::HanashiEncodingImpl, harmony::HarmonyEncodingImpl},
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
    Hanashi(HanashiEncodingImpl),
    Harmony(HarmonyEncodingImpl),
}

impl Encoding {
    /// Decode one generated token without allocating a one-element vector.
    pub fn decode_token(
        &mut self,
        token_id: TokenId,
    ) -> Result<(), Error> {
        dispatch!(self, decode_token, token_id)
    }
}

impl EncodingTrait for Encoding {
    type Config = EncodingConfig;
    type Input = Vec<ChatMessage>;
    type Output = Vec<TokenId>;
    type State = State;
    type Error = Error;

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

    fn supports_tool_calls(&self) -> bool {
        dispatch!(infallible self, supports_tool_calls)
    }

    fn supports_multiple_tool_calls(&self) -> bool {
        dispatch!(infallible self, supports_multiple_tool_calls)
    }
}
