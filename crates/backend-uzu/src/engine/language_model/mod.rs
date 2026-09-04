use std::{
    fs::File,
    io,
    io::BufReader,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    backends::common::{Backend, Context, DeviceCapabilities, Encoder, Kernels, kernel::ContextRingUpdateKernel},
    config::model::{generation::GenerationConfig, language_model::LanguageModelConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::{Decoder, DecoderError},
        sampling::{Sampling, SamplingMethod},
    },
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
    speculators::dflash_tfm::{DFlashSpeculatorLoadError, DFlashTfmSpeculator},
    trie::TrieNode,
};

pub mod state;
pub mod stream;

#[cfg(grammar)]
pub mod grammar;

pub struct LanguageModel<B: Backend> {
    engine: Arc<Engine<B>>,
    decoder: Decoder<B>,
    speculator: Option<DFlashTfmSpeculator<B>>,
    /// QTIP_WEAVER_HEAD=<package dir>: alternative LM head used only by the weaver's candidate scoring
    weaver_head: Option<crate::encodable_block::embedding::Embedding<B>>,
    sampling: Sampling<B>,
    context_ring_update: <B::Kernels as Kernels>::ContextRingUpdateKernel,
    generation_config: GenerationConfig,
    tokenizer: Arc<Tokenizer>,
    #[cfg(grammar)]
    vocab_size: usize,
}

pub struct SuffixForwardBenchmark {
    pub suffix_length: u32,
    pub durations: Box<[Duration]>,
    /// greedy-sampled token ids of the last measured forward (whole-model argmax check)
    pub sampled_tokens: Box<[u32]>,
}

#[derive(Debug, Error)]
pub enum SuffixForwardBenchmarkError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
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
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashSpeculatorLoadError<B>),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

impl<B: Backend> Engine<B> {
    pub fn load_language_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let config: LanguageModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*self.context)?;

        // TODO
        let speculator_path = model_path.join("speculator");
        let speculator_path = speculator_path.exists().then_some(speculator_path);

        let tokenizer = Arc::new(Tokenizer::from_file(model_path.join("tokenizer.json"))?);

        let data_type = DataType::BF16;

        let decoder = Decoder::new(
            self.context.as_ref(),
            &config.decoder_config,
            &weight_loader.tree().subtree("decoder"),
            data_type,
        )?;

        assert!(
            speculator_path.is_none() || decoder.speculation_supported(),
            "attempted to load speculator for a model that doesn't support one"
        );

        let speculator = speculator_path
            .as_deref()
            .map(|speculator_path| DFlashTfmSpeculator::new(speculator_path, self.context.clone()))
            .transpose()?;

        let weaver_head = std::env::var("QTIP_WEAVER_HEAD").ok().map(|head_path| {
            let head_path = Path::new(&head_path);
            let head_config: LanguageModelConfig = serde_json::from_reader(BufReader::new(
                File::open(head_path.join("config.json")).expect("QTIP_WEAVER_HEAD config.json"),
            ))
            .expect("QTIP_WEAVER_HEAD config");
            let head_file = File::open(head_path.join("model.safetensors")).expect("QTIP_WEAVER_HEAD model.safetensors");
            let head_loader = ParameterLoader::new(&head_file, &*self.context).expect("QTIP_WEAVER_HEAD loader");
            let (embedding, _) = crate::encodable_block::embedding::Embedding::new(
                self.context.as_ref(),
                head_config.decoder_config.vocab_size,
                head_config.decoder_config.transformer_config.model_dim,
                &head_config.decoder_config.embedding_config,
                &head_loader.tree().subtree("decoder").subtree("embedding"),
                data_type,
            )
            .expect("QTIP_WEAVER_HEAD embedding");
            eprintln!("weaver head: using the LM head from {}", head_path.display());
            embedding
        });

        let sampling = Sampling::new(data_type, config.decoder_config.vocab_size);

        let context_ring_update = <B::Kernels as Kernels>::ContextRingUpdateKernel::new(&self.context)
            .map_err(EngineLoadLanguageModelError::Backend)?;

        weight_loader.tree().assert_all_tensors_validated()?;

        let generation_config = config.generation_config;

        #[cfg(grammar)]
        let vocab_size = config.decoder_config.vocab_size as usize;

        Ok(LanguageModel {
            engine: self.clone(),
            decoder,
            speculator,
            weaver_head,
            sampling,
            context_ring_update,
            generation_config,
            tokenizer,
            #[cfg(grammar)]
            vocab_size,
        })
    }
}

impl<B: Backend> LanguageModel<B> {
    pub fn max_context_length(&self) -> Option<u32> {
        self.decoder.max_context_length()
    }

    pub fn recommended_context_length(&self) -> Option<u32> {
        let max_context_length = self.max_context_length();

        // TODO: This is not the correct way to do it, there should be a real memory model
        if self.engine.context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
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

            Some(u32::min(max_context_length, platform_recommended_context_length))
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

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    pub fn benchmark_suffix_forwards(
        &self,
        prefix_length: u32,
        suffix_lengths: &[u32],
        warmup_runs: u32,
        measured_runs: u32,
    ) -> Result<Box<[SuffixForwardBenchmark]>, SuffixForwardBenchmarkError<B>> {
        assert!(prefix_length > 0);
        assert!(!suffix_lengths.is_empty());
        assert!(suffix_lengths.iter().all(|&length| length > 0));
        assert!(measured_runs > 0);

        let max_context_length = prefix_length + suffix_lengths.len() as u32 * (warmup_runs + measured_runs);
        let mut state = self.create_empty_state(Some(max_context_length)).map_err(|error| match error {
            state::LanguageModelCreateEmptyStateError::Backend(error) => SuffixForwardBenchmarkError::Backend(error),
        })?;
        let pool = Arc::new(self.engine.context.create_allocation_pool(false));

        for chunk_length in (0..prefix_length).step_by(64).map(|offset| (prefix_length - offset).min(64)) {
            let tokens = vec![1u64; chunk_length as usize];
            let trie = TrieNode::flat(state.transformer_state.context_length() as usize, &tokens, &state.prng);
            let flat_trie = trie.linearize();
            let nodes = flat_trie.token_subtrie_ranges().collect::<Box<[_]>>();
            let batch = BatchTopology::new(&nodes, true);
            let mut encoder =
                Encoder::<B>::new_with_pool_name(&self.engine.context, pool.clone(), Some("suffix prefill"))
                    .map_err(SuffixForwardBenchmarkError::Backend)?;
            let mut token_ids = encoder
                .allocate_constant(tokens.len() * u32::BITS as usize / 8)
                .map_err(SuffixForwardBenchmarkError::Backend)?;
            token_ids.copyin(&tokens.iter().map(|&token| token as u32).collect::<Box<[_]>>());

            state
                .transformer_state
                .prepare(state.transformer_state.context_length(), chunk_length, &self.engine.context)
                .map_err(SuffixForwardBenchmarkError::Backend)?;
            self.decoder.encode(&token_ids, &batch, None, None, &mut state.transformer_state, &mut encoder)?;
            state
                .transformer_state
                .encode_accept(&(0..chunk_length).collect::<Box<[_]>>(), &mut encoder)
                .map_err(SuffixForwardBenchmarkError::Backend)?;
            encoder.end_encoding().submit().wait_until_completed().map_err(SuffixForwardBenchmarkError::Backend)?;
        }

        let mut benchmarks = Vec::with_capacity(suffix_lengths.len());
        for &suffix_length in suffix_lengths {
            let mut durations = Vec::with_capacity(measured_runs as usize);
            let mut last_sampled: Vec<u32> = Vec::new();
            for run in 0..warmup_runs + measured_runs {
                let tokens = vec![1u64; suffix_length as usize];
                let trie = TrieNode::flat(state.transformer_state.context_length() as usize, &tokens, &state.prng);
                let flat_trie = trie.linearize();
                let nodes = flat_trie.token_subtrie_ranges().collect::<Box<[_]>>();
                let batch = BatchTopology::new(&nodes, false);
                let mut encoder =
                    Encoder::<B>::new_with_pool_name(&self.engine.context, pool.clone(), Some("suffix benchmark"))
                        .map_err(SuffixForwardBenchmarkError::Backend)?;
                let mut token_ids = encoder
                    .allocate_constant(tokens.len() * u32::BITS as usize / 8)
                    .map_err(SuffixForwardBenchmarkError::Backend)?;
                token_ids.copyin(&tokens.iter().map(|&token| token as u32).collect::<Box<[_]>>());

                state
                    .transformer_state
                    .prepare(state.transformer_state.context_length(), suffix_length, &self.engine.context)
                    .map_err(SuffixForwardBenchmarkError::Backend)?;
                let output = self.decoder.encode(
                    &token_ids,
                    &batch,
                    Some(0..suffix_length),
                    None,
                    &mut state.transformer_state,
                    &mut encoder,
                )?;
                let logits = output.logits.as_ref().unwrap();
                let sampled = self
                    .sampling
                    .encode(
                        logits,
                        None,
                        None,
                        None,
                        Some(&token_ids),
                        &SamplingMethod::Greedy {},
                        &batch,
                        0..suffix_length,
                        &mut encoder,
                    )
                    .map_err(SuffixForwardBenchmarkError::Backend)?;
                state
                    .transformer_state
                    .encode_accept(&[0], &mut encoder)
                    .map_err(SuffixForwardBenchmarkError::Backend)?;

                let start = Instant::now();
                encoder.end_encoding().submit().wait_until_completed().map_err(SuffixForwardBenchmarkError::Backend)?;
                let duration = start.elapsed();
                assert!(!sampled.as_slice::<u32>().is_empty());
                if run >= warmup_runs {
                    durations.push(duration);
                    last_sampled = sampled.as_slice::<u32>().to_vec();
                }
            }
            benchmarks.push(SuffixForwardBenchmark {
                suffix_length,
                durations: durations.into_boxed_slice(),
                sampled_tokens: last_sampled.into_boxed_slice(),
            });
        }

        Ok(benchmarks.into_boxed_slice())
    }
}
