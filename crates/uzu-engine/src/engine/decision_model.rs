use std::{io, path::Path, sync::Arc, time::Instant};

use shoji::{traits::backend::decision::DecisionProbabilities, types::session::classification::TokenCodecConfig};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    backends::common::{Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    config::{
        model::decision_model::{DecisionConfig, DecisionModelConfig},
        token_codec::AnyTokenCodecConfig,
    },
    data_type::DataType,
    encodable_block::decision::{Decision, DecisionError},
    engine::Engine,
    safetensors::{ParameterLoader, ParameterLoaderError},
    utils::file,
};

#[derive(Debug, Error)]
pub enum DecisionModelError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Input larger than model context size")]
    ContextOverflow,
    #[error("Decision error: {0}")]
    Decision(#[from] DecisionError<B>),
    #[error("Input is empty")]
    EmptyInput,
    #[error("Option count {option_count} is outside 1..={answer_count}")]
    InvalidOptionCount {
        option_count: usize,
        answer_count: usize,
    },
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

pub struct DecisionModel<B: Backend> {
    context: Arc<B::Context>,
    decision: Decision<B>,
    decision_config: DecisionConfig,
    tokenizer: Arc<Tokenizer>,
    token_codec_config: AnyTokenCodecConfig,
}

impl<B: Backend> DecisionModel<B> {
    pub fn decision_config(&self) -> &DecisionConfig {
        &self.decision_config
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    pub fn token_codec_config(&self) -> TokenCodecConfig {
        TokenCodecConfig::from(&self.token_codec_config)
    }

    pub async fn probabilities(
        &self,
        input: &[u64],
        option_count: usize,
    ) -> Result<DecisionProbabilities, DecisionModelError<B>> {
        if input.is_empty() {
            return Err(DecisionModelError::EmptyInput);
        }

        if self
            .decision
            .max_context_length()
            .is_some_and(|max_context_length| input.len() > max_context_length as usize)
        {
            return Err(DecisionModelError::ContextOverflow);
        }

        let answer_count = self.decision_config.codes.len();
        if !(1..=answer_count).contains(&option_count) {
            return Err(DecisionModelError::InvalidOptionCount {
                option_count,
                answer_count,
            });
        }

        let prefill_start = Instant::now();

        let mut command_buffer = self.context.create_command_buffer(None, None).map_err(DecisionModelError::Backend)?;
        let token_ids = command_buffer
            .allocate_constant_from_slice(&input.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>())
            .map_err(DecisionModelError::Backend)?;

        let logits = self.decision.encode(&token_ids, input.len() as u32, &mut command_buffer)?;
        let mut output_buffer = self
            .context
            .create_buffer_gpu(answer_count * DataType::F32.size_in_bytes())
            .map_err(DecisionModelError::Backend)?;
        command_buffer.encode_copy(&logits, &mut output_buffer);
        drop(logits);
        drop(token_ids);

        command_buffer.end_encoding().submit().wait_until_completed().await.map_err(DecisionModelError::Backend)?;

        let prefill_duration = prefill_start.elapsed();

        let temperature = self.decision_config.temperature;
        let logits = self.context.copyout::<f32>(&output_buffer).await.map_err(DecisionModelError::Backend)?;
        let scaled = logits[..option_count].iter().map(|logit| logit / temperature).collect::<Vec<_>>();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = scaled.iter().map(|logit| (logit - max).exp()).collect::<Vec<_>>();
        let sum = exps.iter().sum::<f32>();
        let probabilities = exps.into_iter().map(|value| value / sum).collect();
        Ok(DecisionProbabilities {
            probabilities,
            prefill_duration,
        })
    }
}

impl<B: Backend> Engine<B> {
    pub async fn load_decision_model(
        &self,
        model_path: &Path,
    ) -> Result<DecisionModel<B>, DecisionModelError<B>> {
        let context = self.context.clone();

        let config_file_path = model_path.join("config.json");
        let config_file_content = file::read(config_file_path).await?;
        let config = serde_json::from_slice::<DecisionModelConfig>(&config_file_content)?;

        let tokenizer_file_path = model_path.join("tokenizer.json");
        let tokenizer_file_content = file::read(tokenizer_file_path).await?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_file_content)?;

        let weight_loader = ParameterLoader::new(model_path.join("model.safetensors"), context.as_ref()).await?;
        let decision = Decision::new(
            DataType::BF16,
            DataType::F32,
            &config,
            &weight_loader.tree().subtree("decoder"),
            context.as_ref(),
        )
        .await?;
        weight_loader.assert_all_tensors_read()?;

        Ok(DecisionModel {
            context,
            decision,
            decision_config: config.decision,
            tokenizer: Arc::new(tokenizer),
            token_codec_config: config.token_codec_config,
        })
    }
}
