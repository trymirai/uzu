pub mod chat;

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
}
