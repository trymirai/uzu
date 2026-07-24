use tokenizers::Tokenizer;

mod classification;
mod util;

pub mod chat;

pub use classification::encoding::ClassificationEncoding;

pub trait Encoding {
    type Config;
    type Context;
    type Input;
    type Output;
    type State;
    type Error;

    fn new(
        config: Self::Config,
        context: Self::Context,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn state(&self) -> &Self::State;
    fn reset(&mut self) -> Result<(), Self::Error>;

    fn encode(
        &mut self,
        value: Self::Input,
    ) -> Result<(), Self::Error>;
    fn decode(
        &mut self,
        value: Self::Output,
    ) -> Result<(), Self::Error>;

    fn tokenizer(&self) -> Option<&Tokenizer>;

    fn supports_tool_calls(&self) -> bool;

    fn supports_multiple_tool_calls(&self) -> bool;
}
