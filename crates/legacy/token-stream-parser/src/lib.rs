pub mod extraction;
pub mod framing;
pub mod reduction;
pub mod token_stream;
pub mod types;

use types::Token;

pub trait ParserState {
    fn is_substate_of(
        &self,
        other_state: &Self,
    ) -> bool;

    fn tokens(&self) -> Vec<&Token>;
}

pub trait Parser {
    type Config;
    type Input;
    type Output;
    type State: ParserState;
    type Error;

    fn new(config: Self::Config) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn push(
        &mut self,
        input: &Self::Input,
    ) -> Result<Self::Output, Self::Error>;
    fn state(&self) -> &Self::State;
    fn reset(&mut self);
}
