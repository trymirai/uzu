use std::{fs::File, io, io::BufReader, path::Path, rc::Rc};

use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Context, Kernels,
        kernel::{ContextRingUpdateKernel, TokenCopySampledKernel},
    },
    config::model::{generation::GenerationConfig, language_model::LanguageModelConfig},
    data_type::DataType,
    encodable_block::{
        decoder::{Decoder, DecoderError},
        sampling::{Sampling, SamplingMethod},
    },
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
};

pub mod grammar;
pub mod state;
pub mod stream;

pub struct LanguageModel<B: Backend> {
    context: Rc<B::Context>,
    decoder: Decoder<B>,
    sampling: Sampling<B>,
    token_copy: <B::Kernels as Kernels>::TokenCopySampledKernel,
    context_ring_update: <B::Kernels as Kernels>::ContextRingUpdateKernel,
    generation_config: GenerationConfig,
    vocab_size: usize,
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
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
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
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let data_type = DataType::BF16;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, context.as_ref())?;

        let decoder = Decoder::new(
            context.as_ref(),
            &config.decoder_config,
            &weight_loader.tree().subtree("decoder")?,
            data_type,
        )?;

        let sampling = Sampling::new(data_type, config.decoder_config.vocab_size);

        let token_copy = <B::Kernels as Kernels>::TokenCopySampledKernel::new(&context)
            .map_err(EngineLoadLanguageModelError::Backend)?;
        let context_ring_update = <B::Kernels as Kernels>::ContextRingUpdateKernel::new(&context)
            .map_err(EngineLoadLanguageModelError::Backend)?;

        weight_loader.tree().assert_all_tensors_validated()?;

        let generation_config = config.generation_config;

        let vocab_size = config.decoder_config.vocab_size;

        Ok(LanguageModel {
            context,
            decoder,
            sampling,
            token_copy,
            context_ring_update,
            generation_config,
            vocab_size,
        })
    }
}

impl<B: Backend> LanguageModel<B> {
    pub fn max_context_length(&self) -> Option<usize> {
        self.decoder.max_context_length()
    }

    pub fn recommended_context_length(&self) -> Option<usize> {
        let max_context_length = self.max_context_length();

        // TODO: This is not the correct way to do it, there should be a real memory model
        if self.context.sparse_buffers_supported() {
            // We just assume that all mixers use sparse if it's available to make max context free until it's actually used
            // Currenlty true for all mixers in uzu:
            // - full attention uses sparse if it's available to make max context free until it's actually used
            // - sliding window attention is bound, usually well below the recommended max context size on non-sparse (but can be made to use sparse if we care about it enough)
            // - short conv/mamba2/delta net are constant state size
            max_context_length
        } else if let Some(max_context_length) = max_context_length {
            // If sparse buffers aren't supported and model has finite maximum context length we assume that kv cache is expensive enough that we should probably clamp it to
            // something reasonable-ish for the platform. This is very primitive but works I guess...
            let platform_recommended_context_length = if cfg!(target_os = "ios") {
                8192
            } else {
                16384
            };

            Some(usize::min(max_context_length, platform_recommended_context_length))
        } else {
            // We just assume that unlimited context means constant state size on all mixers and is thus free
            None
        }
    }

    pub fn speculation_supported(&self) -> bool {
        self.decoder.speculation_supported()
    }

    pub fn default_sampling_method(&self) -> SamplingMethod {
        SamplingMethod::Stochastic {
            temperature: self.generation_config.temperature,
            top_k: self.generation_config.top_k,
            top_p: self.generation_config.top_p,
            min_p: self.generation_config.min_p,
            repetition_penalty: self.generation_config.repetition_penalty,
            suffix_repetition_length: self.generation_config.suffix_repetition_length,
        }
    }

    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }
}
