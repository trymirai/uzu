use std::error::Error;

use schemars::JsonSchema;
use thiserror::Error;
use tokenizers::Tokenizer;

#[cfg(grammar_xgrammar)]
mod xgrammar;

#[derive(Debug, Clone)]
pub enum GrammarConfig {
    JsonSchema {
        schema: String,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    },
    Regex {
        pattern: String,
        print_converted_ebnf: bool,
    },
    BuiltinJson,
}

impl GrammarConfig {
    pub fn json_schema(
        schema: String,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    ) -> Self {
        Self::JsonSchema {
            schema,
            any_whitespace,
            indent,
            separators,
            strict_mode,
        }
    }

    pub fn json_schema_simple(schema: String) -> Self {
        Self::JsonSchema {
            schema,
            any_whitespace: true,
            indent: Some(2),
            separators: Some((",".to_string(), ":".to_string())),
            strict_mode: true,
        }
    }

    pub fn regex(
        pattern: String,
        print_converted_ebnf: bool,
    ) -> Self {
        Self::Regex {
            pattern,
            print_converted_ebnf,
        }
    }

    pub fn builtin_json() -> Self {
        Self::BuiltinJson
    }

    pub fn from_json_schema_type<T: JsonSchema>() -> Result<Self, String> {
        let schema_root = schemars::schema_for!(T);
        let schema_str =
            serde_json::to_string(&schema_root).map_err(|e| format!("Failed to serialize schema: {}", e))?;

        Ok(Self::JsonSchema {
            schema: schema_str,
            any_whitespace: true,
            indent: Some(2),
            separators: Some((",".to_string(), ":".to_string())),
            strict_mode: true,
        })
    }

    pub fn from_json_schema_type_with_config<T: JsonSchema>(
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    ) -> Result<Self, String> {
        let schema_root = schemars::schema_for!(T);
        let schema_str =
            serde_json::to_string(&schema_root).map_err(|e| format!("Failed to serialize schema: {}", e))?;

        Ok(Self::JsonSchema {
            schema: schema_str,
            any_whitespace,
            indent,
            separators,
            strict_mode,
        })
    }
}

#[derive(Debug, Error)]
#[error("Grammar error: {0}")]
pub struct GrammarError(Box<dyn Error>);

#[derive(Debug, Error)]
#[error("No grammar backend available")]
pub struct NoGrammarBackend;

pub trait CompiledGrammar {
    fn next_bitmask(&mut self) -> Result<Option<Box<[u32]>>, GrammarError>;

    fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), GrammarError>;

    fn rollback(
        &mut self,
        num_tokens: usize,
    );

    fn is_terminated(&self) -> bool;
}

impl dyn CompiledGrammar {
    #[cfg_attr(not(grammar_xgrammar), allow(unused_variables))]
    pub fn new(
        config: &GrammarConfig,
        tokenizer: &Tokenizer,
        trigger_token_id: Option<u64>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Box<dyn CompiledGrammar>, GrammarError> {
        #[cfg(grammar_xgrammar)]
        return Ok(Box::new(xgrammar::CompiledXGrammar::new(config, tokenizer, trigger_token_id, stop_token_ids)?));

        #[cfg(not(grammar_xgrammar))]
        return Err(GrammarError(Box::new(NoGrammarBackend)));
    }
}
