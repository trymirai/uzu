use std::{fs::File, io, io::BufReader, path::Path, rc::Rc};

use thiserror::Error;

use crate::{
    backends::common::Backend,
    config::model::{generation::GenerationConfig, language_model::LanguageModelConfig},
    data_type::DataType,
    encodable_block::{
        decoder::{Decoder, DecoderError},
        sampling::{Sampling, SamplingMethod, SamplingProcessingOrder},
    },
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
};

pub mod grammar;
pub mod prng;
pub mod state;
pub mod stream;

pub struct LanguageModel<B: Backend> {
    context: Rc<B::Context>,
    decoder: Decoder<B>,
    sampling: Sampling<B>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Error)]
pub enum EngineLoadLanguageModelError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("HeaderLoading error: {0}")]
    HeaderLoading(#[from] HeaderLoadingError),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
}

impl<B: Backend> Engine<B> {
    pub fn load_language_model(
        &self,
        model_path: &Path,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let context = self.context.clone();

        let config: LanguageModelConfig =
            serde_json::from_reader(BufReader::new(File::open(&model_path.join("config.json"))?))?;

        let data_type = DataType::BF16;

        let weights_file = File::open(&model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, context.as_ref())?;

        let decoder = Decoder::new(
            context.as_ref(),
            &config.decoder_config,
            &weight_loader.tree().subtree("decoder")?,
            data_type,
        )?;

        let sampling = Sampling::new(data_type, config.decoder_config.vocab_size);

        weight_loader.tree().assert_all_tensors_validated()?;

        let generation_config = config.generation_config;

        Ok(LanguageModel {
            context,
            decoder,
            sampling,
            generation_config,
        })
    }
}

impl<B: Backend> LanguageModel<B> {
    pub fn default_sampling_method(&self) -> SamplingMethod {
        SamplingMethod::Stochastic {
            temperature: self.generation_config.temperature,
            top_k: self.generation_config.top_k,
            top_p: self.generation_config.top_p,
            min_p: self.generation_config.min_p,
            repetition_penalty: self.generation_config.repetition_penalty,
            suffix_repetition_length: self.generation_config.suffix_repetition_length,
            processing_order: SamplingProcessingOrder::TemperatureThenFilters,
        }
    }

    pub fn default_stop_token_ids(&self) -> &[u64] {
        &self.generation_config.stop_token_ids
    }
}
