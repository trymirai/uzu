use crate::{prelude::Error, session::config::GrammarConfig};

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
    #[allow(unused)] config: Option<GrammarConfig>,
    #[allow(unused)] tokenizer: &tokenizers::Tokenizer,
    #[allow(unused)] stop_token_ids: Option<&[i32]>,
) -> Result<Option<Box<dyn CompiledGrammar>>, Error> {
    #[cfg(grammar_xgrammar)]
    {
        let Some(ref grammar_config) = config else {
            return Ok(None);
        };

        use xgrammar::TokenizerInfo;
        let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, stop_token_ids)
            .map_err(|msg| Error::GrammarError(msg))?;

        use crate::language_model::grammar_xgrammar::CompiledXGrammar;
        let grammar = CompiledXGrammar::from_config(grammar_config, None, &tokenizer_info)?;

        Ok(Some(Box::new(grammar)))
    }

    #[cfg(not(grammar_xgrammar))]
    Ok(None)
}
