use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::session::types::Input;

use super::{AudioError, AudioResult};

pub const DEFAULT_AUDIO_DECODE_CHUNK_FRAMES: usize = 16;

pub trait InputTokenAdapter: Send + Sync {
    fn assemble_tokens(
        &self,
        tokenizer: &Tokenizer,
        input: &Input,
        processed_text: &str,
    ) -> AudioResult<Vec<u64>>;
}

pub trait OutputTokenAdapter: Send + Sync {
    fn decode_for_output_parse(
        &self,
        tokenizer: &Tokenizer,
        generated_tokens: &[u64],
    ) -> AudioResult<String>;

    fn decode_chunk_frames(&self) -> usize {
        DEFAULT_AUDIO_DECODE_CHUNK_FRAMES
    }
}

#[derive(Debug, Default)]
pub struct TextInputTokenAdapter;

impl InputTokenAdapter for TextInputTokenAdapter {
    fn assemble_tokens(
        &self,
        tokenizer: &Tokenizer,
        _input: &Input,
        processed_text: &str,
    ) -> AudioResult<Vec<u64>> {
        let encoding =
            tokenizer.encode(processed_text, false).map_err(|error| AudioError::Tokenizer(error.to_string()))?;

        Ok(encoding.get_ids().iter().map(|&id| id as u64).collect())
    }
}

#[derive(Debug, Default)]
pub struct TextOutputTokenAdapter;

impl OutputTokenAdapter for TextOutputTokenAdapter {
    fn decode_for_output_parse(
        &self,
        tokenizer: &Tokenizer,
        generated_tokens: &[u64],
    ) -> AudioResult<String> {
        let tokens: Vec<u32> = generated_tokens.iter().map(|&token| token as u32).collect();
        tokenizer.decode(&tokens, true).map_err(|error| AudioError::Tokenizer(error.to_string()))
    }
}

#[derive(Clone)]
pub struct TokenAdapters {
    input: Arc<dyn InputTokenAdapter>,
    output: Arc<dyn OutputTokenAdapter>,
}

impl TokenAdapters {
    pub fn new(
        input: Arc<dyn InputTokenAdapter>,
        output: Arc<dyn OutputTokenAdapter>,
    ) -> Self {
        Self {
            input,
            output,
        }
    }

    pub fn input(&self) -> Arc<dyn InputTokenAdapter> {
        self.input.clone()
    }

    pub fn output(&self) -> Arc<dyn OutputTokenAdapter> {
        self.output.clone()
    }
}

impl Default for TokenAdapters {
    fn default() -> Self {
        Self {
            input: Arc::new(TextInputTokenAdapter),
            output: Arc::new(TextOutputTokenAdapter),
        }
    }
}
