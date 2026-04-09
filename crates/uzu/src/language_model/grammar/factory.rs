#[cfg(grammar_xgrammar)]
use xgrammar::TokenizerInfo;

use crate::{language_model::grammar::CompiledGrammar, prelude::Error, session::config::GrammarConfig};

pub struct CompiledGrammarFactory {
    #[cfg(grammar_xgrammar)]
    tokenizer_info: TokenizerInfo,
}

impl CompiledGrammarFactory {
    pub fn new(
        #[allow(unused)] tokenizer: &tokenizers::Tokenizer,
        #[allow(unused)] stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, String> {
        #[cfg(grammar_xgrammar)]
        return Ok(Self {
            tokenizer_info: TokenizerInfo::from_huggingface(&tokenizer, None, stop_token_ids)?,
        });

        #[cfg(not(grammar_xgrammar))]
        Ok(Self {})
    }

    pub fn create(
        &self,
        #[allow(unused)] config: Option<GrammarConfig>,
    ) -> Result<Option<Box<dyn CompiledGrammar>>, Error> {
        #[cfg(grammar_xgrammar)]
        {
            use crate::language_model::grammar::xgram::CompiledXGrammar;

            let Some(ref grammar_config) = config else {
                return Ok(None);
            };
            let grammar = CompiledXGrammar::from_config(grammar_config, None, &self.tokenizer_info)?;
            Ok(Some(Box::new(grammar)))
        }

        #[cfg(not(grammar_xgrammar))]
        Ok(None)
    }
}
