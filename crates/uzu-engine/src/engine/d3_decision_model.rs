use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
    time::Instant,
};

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    },
    config::model::d3_decision::D3DecisionModelConfig,
    encodable_block::d3_decision::{D3Decision, D3DecisionEncodeError, D3DecisionError},
    engine::Engine,
    safetensors::{ParameterLoader, ParameterLoaderError},
};

pub struct D3DecisionModel<B: Backend> {
    context: Arc<B::Context>,
    block: D3Decision<B>,
    config: D3DecisionModelConfig,
    max_context_length: Option<u32>,
}

#[derive(Debug, Error)]
pub enum EngineLoadD3DecisionError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Configuration JSON error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("D3 model error: {0}")]
    D3(#[from] D3DecisionError<B>),
}

impl<B: Backend> Engine<B> {
    pub fn load_d3_decision_model(
        &self,
        model_path: &Path,
    ) -> Result<D3DecisionModel<B>, EngineLoadD3DecisionError<B>> {
        let config: D3DecisionModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        // Reject incompatible semantics before allocating/loading any large weights.
        config.validate().map_err(D3DecisionError::Configuration)?;
        let weights = File::open(model_path.join("model.safetensors"))?;
        let loader = ParameterLoader::new(&weights, self.context.as_ref())?;
        let block = D3Decision::new(self.context.as_ref(), &config, &loader.tree())?;
        loader.assert_all_tensors_read()?;
        let max_context_length = match (block.max_context_length(), config.decision.max_length) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(bound), None) | (None, Some(bound)) => Some(bound),
            (None, None) => None,
        };
        Ok(D3DecisionModel {
            context: self.context.clone(),
            block,
            config,
            max_context_length,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum D3InputError {
    #[error("Input is empty")]
    EmptyInput,
    #[error("Input has {input_tokens} tokens, exceeding context bound {max_tokens}")]
    ContextOverflow {
        input_tokens: usize,
        max_tokens: u32,
    },
    #[error("Token id {token_id} at position {position} is outside vocabulary size {vocab_size}")]
    InvalidToken {
        position: usize,
        token_id: u32,
        vocab_size: u32,
    },
    #[error("Input token count cannot be represented as u32")]
    TokenCountOverflow,
}

fn validate_input(
    input: &[u32],
    vocab_size: u32,
    max_context_length: Option<u32>,
) -> Result<u32, D3InputError> {
    if input.is_empty() {
        return Err(D3InputError::EmptyInput);
    }
    let sequence_length = u32::try_from(input.len()).map_err(|_| D3InputError::TokenCountOverflow)?;
    if let Some(max_tokens) = max_context_length
        && sequence_length > max_tokens
    {
        return Err(D3InputError::ContextOverflow {
            input_tokens: input.len(),
            max_tokens,
        });
    }
    if let Some((position, &token_id)) = input.iter().enumerate().find(|(_, token_id)| **token_id >= vocab_size) {
        return Err(D3InputError::InvalidToken {
            position,
            token_id,
            vocab_size,
        });
    }
    Ok(sequence_length)
}

#[derive(Debug, Error)]
pub enum D3DecisionInferenceError<B: Backend> {
    #[error(transparent)]
    Input(#[from] D3InputError),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("D3 encoding error: {0}")]
    Encode(#[from] D3DecisionEncodeError<B>),
    #[error("D3 output contains nonfinite values")]
    NonfiniteOutput,
}

pub struct D3DecisionOutput {
    pub logits: Vec<f32>,
    pub hidden: Option<Vec<f32>>,
    pub forward_ms: f64,
    pub input_tokens: usize,
    pub peak_memory_bytes: Option<usize>,
}

impl<B: Backend> D3DecisionModel<B> {
    pub fn model_dim(&self) -> u32 {
        self.config.transformer_config.model_dim
    }
    pub fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }
    pub fn num_codes(&self) -> usize {
        self.config.decision.codes.len()
    }
    pub fn device_name(&self) -> Option<&str> {
        self.context.device_name()
    }

    pub fn infer(
        &self,
        token_ids: &[u32],
        return_hidden: bool,
    ) -> Result<D3DecisionOutput, D3DecisionInferenceError<B>> {
        let sequence_length = validate_input(token_ids, self.vocab_size(), self.max_context_length)?;
        let start = Instant::now();
        // This state is fresh for every call and lives through GPU completion.
        // No prompt prefix, KV cache, or DeltaNet recurrent state is reused.
        let mut state = self
            .block
            .create_empty_state(sequence_length, self.context.as_ref())
            .map_err(D3DecisionInferenceError::Backend)?;
        let mut command_buffer =
            self.context.create_command_buffer(Some("D3 forward"), None).map_err(D3DecisionInferenceError::Backend)?;
        let input =
            command_buffer.allocate_constant_from_slice(token_ids).map_err(D3DecisionInferenceError::Backend)?;
        let output = self.block.encode(&input, sequence_length, &mut state, &mut command_buffer)?;
        let mut logits_buffer = self
            .context
            .create_buffer(self.num_codes() * size_of::<f32>())
            .map_err(D3DecisionInferenceError::Backend)?;
        command_buffer.encode_copy(&output.logits, &mut logits_buffer);
        let mut hidden_buffer = if return_hidden {
            Some(
                self.context
                    .create_buffer(self.model_dim() as usize * size_of::<f32>())
                    .map_err(D3DecisionInferenceError::Backend)?,
            )
        } else {
            None
        };
        if let Some(hidden_buffer) = &mut hidden_buffer {
            command_buffer.encode_copy(&output.hidden, hidden_buffer);
        }
        // Metal's constant allocator must have no live allocation handles when
        // encoding ends; the executable retains its backing allocation itself.
        drop(input);
        command_buffer.end_encoding().submit().wait_until_completed().map_err(D3DecisionInferenceError::Backend)?;
        let forward_ms = start.elapsed().as_secs_f64() * 1000.0;
        let logits = logits_buffer.copyout::<f32>();
        let hidden = hidden_buffer.as_ref().map(|buffer| buffer.copyout::<f32>());
        if logits.iter().any(|value| !value.is_finite())
            || hidden.as_ref().is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(D3DecisionInferenceError::NonfiniteOutput);
        }
        Ok(D3DecisionOutput {
            logits,
            hidden,
            forward_ms,
            input_tokens: token_ids.len(),
            peak_memory_bytes: self.context.peak_memory_usage(),
        })
    }
}

#[cfg(test)]
#[path = "../../unit/engine/d3_decision_model_test.rs"]
mod tests;
