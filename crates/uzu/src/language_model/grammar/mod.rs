mod factory;
pub(crate) mod xgram;

pub use factory::CompiledGrammarFactory;

use crate::prelude::Error;

pub trait CompiledGrammar {
    fn next_bitmask(&mut self) -> Result<Option<Box<[u32]>>, Error>;

    fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), Error>;

    fn rollback(
        &mut self,
        num_tokens: usize,
    );

    fn is_terminated(&self) -> bool;
}
