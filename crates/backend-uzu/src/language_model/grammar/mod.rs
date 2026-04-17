mod xgram;

use tokenizers::Tokenizer;

use crate::session::{config::GrammarConfig, types::Error};

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

pub fn create_compiled_grammar(
    #[allow(unused)] config: &GrammarConfig,
    #[allow(unused)] tokenizer: &Tokenizer,
    #[allow(unused)] stop_token_ids: Option<&[i32]>,
) -> Result<Box<dyn CompiledGrammar>, Error> {
    #[cfg(grammar_xgrammar)]
    {
        use xgrammar::TokenizerInfo;
        let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, stop_token_ids)
            .map_err(|msg| Error::GrammarError(msg))?;

        use xgram::CompiledXGrammar;
        let grammar = CompiledXGrammar::from_config(config, None, &tokenizer_info)?;

        Ok(Box::new(grammar))
    }

    #[cfg(not(grammar_xgrammar))]
    Err(Error::GrammarNoBackendAvailable)
}
