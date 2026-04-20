use std::{collections::BTreeSet, sync::LazyLock};

use regex::Regex;

use super::*;
use crate::{
    array::{Array, ArrayContextExt},
    backends::common::{Allocation, Buffer},
    config::TtsMessageProcessorConfig,
    encodable_block::SamplingInputs,
    session::types::{TtsModelConfigError, TtsPromptConfigError},
};

pub(super) struct FishAudioTextDecoderRuntime<B: Backend> {
    slow_runner: TokenDecoderRunner<B>,
    fast_runner: TokenDecoderRunner<B>,
    semantic_bridge: FishAudioSemanticBridge<B>,
    runtime_config: TextDecoderRuntimeConfig,
    semantic_token_begin_id: i64,
    semantic_token_end_id: i64,
    semantic_cardinality: usize,
    im_end_token_id: i64,
    codebook_size: usize,
    num_codebooks: usize,
    slow_model_dim: usize,
    fast_model_dim: usize,
    max_seq_len: usize,
    scale_codebook_embeddings: bool,
    fast_vocab_limit: usize,
    apply_semantic_sampling_mask: bool,
    semantic_sampling_mask_row: Box<[u32]>,
    semantic_sampling_mask_without_im_end_row: Box<[u32]>,
    current_codes_scratch: Vec<u32>,
    instrumentation: RunnerInstrumentation,
}

struct FishAudioSemanticBridge<B: Backend> {
    embedding_rows_sum: <B::Kernels as Kernels>::EmbeddingRowsSumKernel,
    projection: <B::Kernels as ManualKernels>::MatmulKernel,
    codebook_embeddings: Allocation<B>,
    codebook_token_indices: Array<B>,
    projection_weights: Option<Allocation<B>>,
    num_codebooks: usize,
    codebook_size: usize,
    slow_model_dim: usize,
    fast_model_dim: usize,
}

impl<B: Backend> FishAudioSemanticBridge<B> {
    fn load(
        context: &B::Context,
        parameter_tree: &crate::parameters::ParameterTree<B::Context>,
        config: &crate::config::FishAudioTextDecoderConfig,
        data_type: DataType,
    ) -> Result<Self, Error> {
        let codebook_embeddings = load_float_tensor_allocation(
            parameter_tree,
            "text_decoder.codebook_embeddings.weights",
            [
                config.codebook_size.checked_mul(config.num_codebooks).ok_or(Error::UnableToLoadConfig)?,
                config.slow_model_dim,
            ],
            data_type,
        )?;
        let fast_model_projection = if config.fast_model_projection_config.is_some() {
            Some(load_float_tensor_allocation(
                parameter_tree,
                "text_decoder.fast_model_projection.weights",
                [config.fast_model_dim, config.slow_model_dim],
                data_type,
            )?)
        } else {
            None
        };

        Self::new(
            context,
            data_type,
            codebook_embeddings,
            fast_model_projection,
            config.num_codebooks,
            config.codebook_size,
            config.slow_model_dim,
            config.fast_model_dim,
        )
    }

    fn new(
        context: &B::Context,
        data_type: DataType,
        codebook_embeddings: Allocation<B>,
        fast_model_projection: Option<Allocation<B>>,
        num_codebooks: usize,
        codebook_size: usize,
        slow_model_dim: usize,
        fast_model_dim: usize,
    ) -> Result<Self, Error> {
        let embedding_rows_sum = <B::Kernels as Kernels>::EmbeddingRowsSumKernel::new(context, data_type)
            .map_err(unable_to_create_context)?;
        let projection = <<B::Kernels as ManualKernels>::MatmulKernel as MatmulKernel>::new(context, data_type)
            .map_err(unable_to_create_context)?;
        let codebook_token_indices = context.create_array_zeros(&[num_codebooks], DataType::U32, "codebook_token_ids");

        Ok(Self {
            embedding_rows_sum,
            projection,
            codebook_embeddings,
            codebook_token_indices,
            projection_weights: fast_model_projection,
            num_codebooks,
            codebook_size,
            slow_model_dim,
            fast_model_dim,
        })
    }
}

fn load_float_tensor_allocation<B: Backend>(
    parameter_tree: &crate::parameters::ParameterTree<B::Context>,
    key: &str,
    expected_shape: [usize; 2],
    target_data_type: DataType,
) -> Result<Allocation<B>, Error> {
    let leaf = parameter_tree.leaf(key).map_err(|_| Error::UnableToLoadWeights)?;
    if leaf.shape() != expected_shape {
        return Err(Error::UnableToLoadConfig);
    }
    if leaf.data_type() != target_data_type {
        return Err(TtsModelConfigError::FishAudioTensorDataTypeMismatch {
            key: key.into(),
            expected: target_data_type,
            actual: leaf.data_type(),
        }
        .into());
    }
    leaf.read_allocation().map_err(|_| Error::UnableToLoadWeights)
}

fn validate_fishaudio_decoder_contract(
    num_codebooks: usize,
    codebook_size: usize,
    max_seq_len: usize,
    semantic_token_begin_id: i64,
    semantic_token_end_id: i64,
    audio_num_codebooks: usize,
    audio_codec_cardinality: usize,
    audio_semantic_cardinality: usize,
) -> Result<usize, Error> {
    if num_codebooks <= 1 || codebook_size == 0 || max_seq_len == 0 {
        return Err(TtsModelConfigError::FishAudioDecoderCoreParametersInvalid {
            num_codebooks,
            codebook_size,
            max_seq_len,
        }
        .into());
    }
    if num_codebooks != audio_num_codebooks {
        return Err(TtsModelConfigError::FishAudioDecoderNumCodebooksMismatch {
            decoder_num_codebooks: num_codebooks,
            audio_num_codebooks,
        }
        .into());
    }
    if audio_codec_cardinality == 0 || audio_codec_cardinality > codebook_size {
        return Err(TtsModelConfigError::FishAudioDecoderCodebookSizeTooSmall {
            codebook_size,
            audio_codec_cardinality,
        }
        .into());
    }
    if semantic_token_begin_id > semantic_token_end_id {
        return Err(TtsModelConfigError::FishAudioSemanticTokenRangeInvalid {
            begin: semantic_token_begin_id,
            end: semantic_token_end_id,
        }
        .into());
    }

    let semantic_cardinality = usize::try_from(semantic_token_end_id - semantic_token_begin_id + 1).map_err(|_| {
        TtsModelConfigError::FishAudioSemanticTokenRangeOverflow {
            begin: semantic_token_begin_id,
            end: semantic_token_end_id,
        }
    })?;
    if semantic_cardinality == 0 || semantic_cardinality != audio_semantic_cardinality {
        return Err(TtsModelConfigError::FishAudioSemanticCodecCardinalityMismatch {
            semantic_cardinality,
            audio_semantic_cardinality,
        }
        .into());
    }

    Ok(semantic_cardinality)
}

static MESSAGE_FIELD_REF_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"message\.([A-Za-z_][A-Za-z0-9_]*)").expect("message field regex"));

pub(super) fn validate_fishaudio_message_processor_config(config: &TtsMessageProcessorConfig) -> Result<(), Error> {
    let builtin_fields =
        ["role", "content", "reasoning_content", "speaker_id", "style"].into_iter().collect::<BTreeSet<_>>();
    let referenced_fields = MESSAGE_FIELD_REF_REGEX
        .captures_iter(config.prompt_template.as_str())
        .filter_map(|capture| capture.get(1).map(|field| field.as_str().to_owned()))
        .collect::<BTreeSet<_>>();

    for field in referenced_fields {
        if !builtin_fields.contains(field.as_str()) && !config.default_message_fields.contains_key(field.as_str()) {
            return Err(TtsPromptConfigError::MissingDefaultMessageField {
                field: field.into_boxed_str(),
            }
            .into());
        }
    }

    Ok(())
}

pub(super) fn build_fishaudio_text_decoder_runtime<B: Backend>(
    config: &crate::config::FishAudioTextDecoderConfig,
    audio: &AudioGenerationContext<B>,
    model_path: &Path,
    runtime_config: &TextDecoderRuntimeConfig,
) -> Result<Box<dyn SemanticDecoderBackend>, Error> {
    let audio_semantic_cardinality =
        audio.runtime().config().semantic_codec_cardinality().ok_or(Error::UnableToLoadConfig)?;
    let semantic_cardinality = validate_fishaudio_decoder_contract(
        config.num_codebooks,
        config.codebook_size,
        config.max_seq_len,
        config.semantic_token_begin_id,
        config.semantic_token_end_id,
        audio.num_codebooks(),
        audio.codec_cardinality(),
        audio_semantic_cardinality,
    )?;
    if !config.slow_readout_config.is_full_precision() {
        return Err(Error::UnableToLoadConfig);
    }
    if !config.fast_readout_config.is_full_precision() {
        return Err(Error::UnableToLoadConfig);
    }
    if config.slow_model_config.model_dim != config.slow_model_dim {
        return Err(Error::UnableToLoadConfig);
    }
    if config.fast_model_config.model_dim != config.fast_model_dim {
        return Err(Error::UnableToLoadConfig);
    }
    if config.short_logits_size == 0 {
        return Err(Error::UnableToLoadConfig);
    }
    let semantic_sampling_mask_row = build_semantic_sampling_mask_row(
        config.vocab_size,
        config.semantic_token_begin_id,
        config.semantic_token_end_id,
        config.im_end_token_id,
    )?;
    let mut semantic_sampling_mask_without_im_end_row = semantic_sampling_mask_row.to_vec();
    clear_token_in_sampling_mask(&mut semantic_sampling_mask_without_im_end_row, config.im_end_token_id)?;

    let slow_transformer_config = config.slow_model_config.clone();
    let fast_transformer_config = config.fast_model_config.clone();

    let slow_inner_config = InnerModelConfig {
        embedding_config: config.slow_embeddings_config.to_text_decoder_embedding_config(),
        transformer_config: slow_transformer_config,
        vocab_size: config.vocab_size,
    };
    let fast_inner_config = InnerModelConfig {
        embedding_config: config.fast_embeddings_config.to_text_decoder_embedding_config(),
        transformer_config: fast_transformer_config,
        vocab_size: config.codebook_size,
    };

    let slow_decoder_config = Rc::new(slow_inner_config.to_decoder_config().map_err(|_| Error::UnableToLoadConfig)?);
    let fast_decoder_config = Rc::new(fast_inner_config.to_decoder_config().map_err(|_| Error::UnableToLoadConfig)?);
    let text_decoder_context = B::Context::new().map_err(unable_to_create_context)?;

    let slow_runner = TokenDecoderRunner::new_with_context(
        text_decoder_context.clone(),
        model_path,
        slow_decoder_config,
        "text_decoder.transformer_slow",
        "text_decoder.embeddings_slow",
        "text_decoder.readout_slow",
        runtime_config,
    )?;

    let fast_runner = TokenDecoderRunner::new_with_context(
        text_decoder_context.clone(),
        model_path,
        fast_decoder_config,
        "text_decoder.transformer_fast",
        "text_decoder.embeddings_fast",
        "text_decoder.readout_fast",
        runtime_config,
    )?;

    let activation_data_type = slow_runner.activation_data_type();
    let fast_data_type = fast_runner.activation_data_type();
    if fast_data_type != activation_data_type {
        return Err(Error::UnableToLoadConfig);
    }

    let weights_path = model_path.join("model.safetensors");
    let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
    let loader =
        ParameterLoader::new(&weights_file, text_decoder_context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
    let root_weights = loader.tree();

    let apply_semantic_sampling_mask =
        runtime_config.force_semantic_sampling_mask.unwrap_or(!matches!(activation_data_type, DataType::F32));

    let semantic_bridge =
        FishAudioSemanticBridge::load(text_decoder_context.as_ref(), &root_weights, config, activation_data_type)?;

    Ok(Box::new(FishAudioTextDecoderRuntime::<B> {
        slow_runner,
        fast_runner,
        semantic_bridge,
        runtime_config: runtime_config.clone(),
        semantic_token_begin_id: config.semantic_token_begin_id,
        semantic_token_end_id: config.semantic_token_end_id,
        semantic_cardinality,
        im_end_token_id: config.im_end_token_id,
        codebook_size: config.codebook_size,
        num_codebooks: config.num_codebooks,
        slow_model_dim: config.slow_model_dim,
        fast_model_dim: config.fast_model_dim,
        max_seq_len: config.max_seq_len,
        scale_codebook_embeddings: config.scale_codebook_embeddings,
        fast_vocab_limit: config.short_logits_size.min(config.codebook_size),
        apply_semantic_sampling_mask,
        semantic_sampling_mask_row,
        semantic_sampling_mask_without_im_end_row: semantic_sampling_mask_without_im_end_row.into_boxed_slice(),
        current_codes_scratch: vec![0_u32; config.num_codebooks],
        instrumentation: RunnerInstrumentation::default(),
    }))
}

impl<B: Backend> FishAudioTextDecoderRuntime<B> {
    fn generate_semantic_tokens_internal(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        mut on_frame: Option<&mut dyn FnMut(&[u32]) -> Result<(), Error>>,
    ) -> Result<AudioTokenGrid, Error> {
        if text_tokens.is_empty() {
            return AudioTokenGrid::new(
                Vec::new().into_boxed_slice(),
                1,
                self.num_codebooks,
                0,
                vec![0].into_boxed_slice(),
            )
            .map_err(Error::from);
        }
        if text_tokens.len() >= self.max_seq_len {
            return Err(Error::GenerateFailed);
        }

        if codec_cardinality == 0 || codec_cardinality > self.codebook_size {
            return Err(Error::UnableToLoadConfig);
        }
        let semantic_token_upper_bound = self.semantic_cardinality;
        let residual_token_upper_bound = codec_cardinality;
        if semantic_token_upper_bound == 0 || residual_token_upper_bound == 0 {
            return Err(Error::UnableToLoadConfig);
        }

        self.reset_generation_state();
        let mut sampling = TextSamplingState::from_config(seed, &self.runtime_config.sampling);
        let mut current_semantic_token = self.decode_initial_semantic_token(text_tokens, &mut sampling)?;
        let post_scale = if self.scale_codebook_embeddings {
            Some(1.0 / ((self.num_codebooks + 1) as f32).sqrt())
        } else {
            None
        };

        let mut max_new_tokens = self.max_seq_len.saturating_sub(text_tokens.len());
        max_new_tokens = max_new_tokens.min(max_semantic_frames.max(1));
        if let Some(limit) = self.runtime_config.max_new_tokens_override {
            max_new_tokens = max_new_tokens.min(limit);
        }
        let mut by_codebook =
            (0..self.num_codebooks).map(|_| Vec::<u32>::with_capacity(max_new_tokens)).collect::<Vec<_>>();
        if self.current_codes_scratch.len() != self.num_codebooks {
            self.current_codes_scratch = vec![0_u32; self.num_codebooks];
        }

        self.prepare_fast_runner_masks(residual_token_upper_bound)?;

        for _step in 0..max_new_tokens {
            if current_semantic_token as i64 == self.im_end_token_id {
                break;
            }
            let first_code = semantic_token_to_code(
                current_semantic_token,
                self.semantic_token_begin_id,
                self.semantic_token_end_id,
                semantic_token_upper_bound,
            );
            by_codebook[0].push(first_code);
            self.current_codes_scratch[0] = first_code;

            current_semantic_token = self.decode_frame_merged(
                current_semantic_token,
                first_code,
                residual_token_upper_bound,
                post_scale,
                &mut sampling,
                &mut by_codebook,
            )?;

            if let Some(callback) = on_frame.as_mut() {
                callback(&self.current_codes_scratch)?;
            }
        }

        self.instrumentation = self.slow_runner.take_instrumentation();
        let fast = self.fast_runner.take_instrumentation();
        self.instrumentation.command_buffers_submitted += fast.command_buffers_submitted;
        self.instrumentation.host_waits += fast.host_waits;

        self.finish_semantic_grid(&by_codebook)
    }

    fn reset_generation_state(&mut self) {
        self.instrumentation = RunnerInstrumentation::default();
        self.slow_runner.reset();
        self.slow_runner.clear_instrumentation();
        self.fast_runner.clear_instrumentation();
    }

    fn decode_initial_semantic_token(
        &mut self,
        text_tokens: &[u64],
        sampling: &mut TextSamplingState,
    ) -> Result<u64, Error> {
        let slow_runner = &mut self.slow_runner;
        let initial_sampling_row = if self.apply_semantic_sampling_mask {
            if self.runtime_config.min_frames_before_im_end > 0 {
                self.semantic_sampling_mask_without_im_end_row.as_ref()
            } else {
                self.semantic_sampling_mask_row.as_ref()
            }
        } else {
            &[]
        };
        let prefill_step_size = text_decoder_prefill_step_size(&self.runtime_config, self.max_seq_len);
        if text_tokens.len() > prefill_step_size {
            for chunk in text_tokens[..text_tokens.len() - prefill_step_size].chunks(prefill_step_size) {
                slow_runner.prefill_without_sampling(chunk)?;
            }
        }
        let prefill_tail_start = text_tokens.len().saturating_sub(prefill_step_size);
        let prefill_tail = &text_tokens[prefill_tail_start..];
        let prefill_mask = if initial_sampling_row.is_empty() {
            None
        } else {
            Some(expand_token_mask_for_sampling_row(initial_sampling_row, prefill_tail.len())?)
        };
        slow_runner.decode_next_token_with_hidden_capture(
            prefill_tail,
            EmbeddingInjection::None,
            sampling,
            prefill_mask.as_deref(),
        )
    }

    fn prepare_fast_runner_masks(
        &mut self,
        residual_token_upper_bound: usize,
    ) -> Result<(), Error> {
        let fast_vocab_limit = self.fast_vocab_limit.min(residual_token_upper_bound);
        if fast_vocab_limit == 0 {
            return Err(Error::UnableToLoadConfig);
        }
        self.fast_runner.prepare_single_token_vocab_mask(fast_vocab_limit)?;
        self.fast_runner.prepare_two_token_vocab_mask(fast_vocab_limit)?;
        Ok(())
    }

    fn finish_semantic_grid(
        &self,
        by_codebook: &[Vec<u32>],
    ) -> Result<AudioTokenGrid, Error> {
        let frames = by_codebook.first().map_or(0, Vec::len);
        let mut tokens = Vec::with_capacity(self.num_codebooks * frames);
        for codebook_tokens in by_codebook {
            if codebook_tokens.len() != frames {
                return Err(Error::GenerateFailed);
            }
            tokens.extend_from_slice(codebook_tokens);
        }
        AudioTokenGrid::new(tokens.into_boxed_slice(), 1, self.num_codebooks, frames, vec![frames].into_boxed_slice())
            .map_err(Error::from)
    }

    fn encode_project_slow_hidden_to_fast_on(
        context: &B::Context,
        semantic_bridge: &mut FishAudioSemanticBridge<B>,
        slow_hidden_capture: &Allocation<B>,
        output_embedding: &mut Allocation<B>,
        slow_model_dim: usize,
        fast_model_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), Error> {
        if let Some(weights) = semantic_bridge.projection_weights.as_ref() {
            if semantic_bridge.fast_model_dim != fast_model_dim || semantic_bridge.slow_model_dim != slow_model_dim {
                return Err(Error::GenerateFailed);
            }
            semantic_bridge.projection.encode(
                context,
                MatmulArguments {
                    a: slow_hidden_capture,
                    b: weights,
                    ab_scale: 1.0,
                    c: MatmulArgumentC::None,
                    d: output_embedding,
                    batch_dim: 1,
                    input_dim: slow_model_dim as u32,
                    output_dim: fast_model_dim as u32,
                },
                encoder,
            );
            return Ok(());
        }

        if slow_model_dim != fast_model_dim {
            return Err(Error::UnableToLoadConfig);
        }

        debug_assert_eq!(slow_hidden_capture.as_buffer_range().1.len(), output_embedding.as_buffer_range().1.len());
        encoder.encode_copy_allocation(slow_hidden_capture, output_embedding);
        Ok(())
    }

    /// Encode the codebook sum reading residual
    /// codes directly from a GPU buffer instead of from CPU memory.
    ///
    /// `first_code` is written into slot 0 of the staging buffer from CPU.
    /// Slots 1..num_codebooks are filled by a GPU blit from
    /// `gpu_tokens_buffer` at the given byte offset.
    fn encode_slow_codebook_sum_from_gpu_tokens_on(
        semantic_bridge: &mut FishAudioSemanticBridge<B>,
        slow_sum_embedding: &mut Allocation<B>,
        first_code: u32,
        source_runner: &TokenDecoderRunner<B>,
        source_slot: usize,
        num_codebooks: usize,
        codebook_size: usize,
        slow_model_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), Error> {
        if num_codebooks == 0 {
            return Err(Error::UnableToLoadConfig);
        }

        // Write first_code at position 0 from CPU.
        {
            if semantic_bridge.num_codebooks != num_codebooks {
                return Err(Error::GenerateFailed);
            }
            let (buffer, range) = semantic_bridge.codebook_token_indices.as_buffer_range();
            unsafe {
                let ptr = (buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start) as *mut u32;
                *ptr = first_code;
            }
        }

        // Blit residual codes from GPU: async_chain_results[0..N-2] -> codebook_token_indices[1..N-1]
        let residual_count = num_codebooks.saturating_sub(1);
        if residual_count > 0 {
            let dst_byte_offset = std::mem::size_of::<u32>(); // skip slot 0
            let token_indices =
                semantic_bridge.codebook_token_indices.view_at_offset(dst_byte_offset, &[residual_count]);
            source_runner.copy_async_chain_results_to_on(encoder, source_slot, &token_indices, residual_count)?;
        }

        let total_vocab = num_codebooks.checked_mul(codebook_size).ok_or(Error::GenerateFailed)?;
        let total_vocab_u32 = u32::try_from(total_vocab).map_err(|_| Error::GenerateFailed)?;
        let num_codebooks_u32 = u32::try_from(num_codebooks).map_err(|_| Error::GenerateFailed)?;
        let slow_model_dim_u32 = u32::try_from(slow_model_dim).map_err(|_| Error::GenerateFailed)?;
        let codebook_size_u32 = u32::try_from(codebook_size).map_err(|_| Error::GenerateFailed)?;

        let codebook_token_indices = semantic_bridge.codebook_token_indices.allocation();
        let codebook_embeddings = &semantic_bridge.codebook_embeddings;
        if semantic_bridge.num_codebooks != num_codebooks
            || semantic_bridge.codebook_size != codebook_size
            || semantic_bridge.slow_model_dim != slow_model_dim
        {
            return Err(Error::GenerateFailed);
        }
        semantic_bridge.embedding_rows_sum.encode(
            codebook_token_indices,
            codebook_embeddings,
            slow_sum_embedding,
            num_codebooks_u32,
            total_vocab_u32,
            slow_model_dim_u32,
            codebook_size_u32,
            encoder,
        );
        Ok(())
    }

    /// Merged fast+slow frame: encodes all fast runner forward passes, the
    /// codebook sum (reading GPU tokens directly), and the slow runner
    /// forward pass into a **single** command buffer submission.
    ///
    /// Returns the next semantic token.
    fn decode_frame_merged(
        &mut self,
        current_semantic_token: u64,
        first_code: u32,
        residual_token_upper_bound: usize,
        post_scale: Option<f32>,
        sampling: &mut TextSamplingState,
        by_codebook: &mut [Vec<u32>],
    ) -> Result<u64, Error> {
        let fast_vocab_limit = self.fast_vocab_limit.min(residual_token_upper_bound);
        let followup_count = self.num_codebooks.saturating_sub(2);

        self.fast_runner.reset();

        let context = Rc::clone(self.fast_runner.context());
        let mut shared_encoder = Encoder::new(context.as_ref()).map_err(unable_to_create_context)?;

        let mut pending_sampling_inputs = Vec::new();
        let total_fast_count = self.encode_fast_passes_on(
            &mut shared_encoder,
            first_code,
            fast_vocab_limit,
            followup_count,
            sampling,
            &mut pending_sampling_inputs,
        )?;
        self.encode_slow_pass_on(
            &mut shared_encoder,
            current_semantic_token,
            first_code,
            post_scale,
            sampling,
            by_codebook,
            &mut pending_sampling_inputs,
        )?;
        let submit_result = self.slow_runner.submit_and_wait_command_buffer(shared_encoder);
        submit_result?;

        for pass in 0..total_fast_count {
            let sampled = self.fast_runner.read_async_chain_result(pass)?;
            let codebook_index = pass + 1;
            let clamped = (sampled as usize).min(residual_token_upper_bound.saturating_sub(1));
            let clamped = u32::try_from(clamped).map_err(|_| Error::GenerateFailed)?;
            by_codebook[codebook_index].push(clamped);
            self.current_codes_scratch[codebook_index] = clamped;
        }

        let slow_token = u64::from(self.slow_runner.read_async_chain_result(0)?);
        Ok(slow_token)
    }

    /// Encode the fast runner's initial + followup passes onto the shared
    /// command buffer.
    fn encode_fast_passes_on(
        &mut self,
        encoder: &mut Encoder<B>,
        first_code: u32,
        fast_vocab_limit: usize,
        followup_count: usize,
        sampling: &mut TextSamplingState,
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
    ) -> Result<usize, Error> {
        let (slow_runner, fast_runner, semantic_bridge) =
            (&mut self.slow_runner, &mut self.fast_runner, &mut self.semantic_bridge);
        let slow_hidden_capture = &slow_runner.single_hidden_capture;
        let slow_model_dim = self.slow_model_dim;
        let fast_model_dim = self.fast_model_dim;

        let mut pre_projection = |runner: &mut TokenDecoderRunner<B>, encoder: &mut Encoder<B>| {
            let runner_context = Rc::clone(runner.context());
            Self::encode_project_slow_hidden_to_fast_on(
                runner_context.as_ref(),
                semantic_bridge,
                slow_hidden_capture,
                &mut runner.single_override_embedding,
                slow_model_dim,
                fast_model_dim,
                encoder,
            )
        };

        fast_runner.encode_first_and_followup_tokens_on(
            encoder,
            &[0, u64::from(first_code)],
            EmbeddingInjection::OverrideFirstRowInternal,
            Some(&mut pre_projection),
            followup_count,
            Some(fast_vocab_limit),
            sampling,
            pending_sampling_inputs,
        )
    }

    /// Encode the slow runner's forward pass onto the shared command buffer.
    fn encode_slow_pass_on(
        &mut self,
        encoder: &mut Encoder<B>,
        current_semantic_token: u64,
        first_code: u32,
        post_scale: Option<f32>,
        sampling: &mut TextSamplingState,
        by_codebook: &[Vec<u32>],
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
    ) -> Result<(), Error> {
        let (slow_runner, fast_runner, semantic_bridge) =
            (&mut self.slow_runner, &mut self.fast_runner, &mut self.semantic_bridge);
        let num_codebooks = self.num_codebooks;
        let codebook_size = self.codebook_size;
        let slow_model_dim = self.slow_model_dim;

        let sampled_frames = by_codebook[0].len();
        let slow_sampling_mask = if self.apply_semantic_sampling_mask {
            if sampled_frames < self.runtime_config.min_frames_before_im_end {
                Some(self.semantic_sampling_mask_without_im_end_row.as_ref())
            } else {
                Some(self.semantic_sampling_mask_row.as_ref())
            }
        } else {
            None
        };

        let mut pre_codebook_sum = |runner: &mut TokenDecoderRunner<B>, encoder: &mut Encoder<B>| {
            Self::encode_slow_codebook_sum_from_gpu_tokens_on(
                semantic_bridge,
                &mut runner.single_override_embedding,
                first_code,
                fast_runner,
                0,
                num_codebooks,
                codebook_size,
                slow_model_dim,
                encoder,
            )
        };

        slow_runner.encode_next_step_on(
            encoder,
            &[current_semantic_token],
            EmbeddingInjection::AddPreloaded {
                post_scale,
            },
            sampling,
            slow_sampling_mask,
            true, // capture_hidden
            Some(&mut pre_codebook_sum),
            pending_sampling_inputs,
        )
    }

    fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        std::mem::take(&mut self.instrumentation)
    }
}

impl<B: Backend> SemanticDecoderBackend for FishAudioTextDecoderRuntime<B> {
    fn generate_semantic_tokens(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        self.generate_semantic_tokens_internal(text_tokens, codec_cardinality, seed, max_semantic_frames, None)
    }

    fn generate_semantic_tokens_with_callback(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<(), Error>,
    ) -> Result<AudioTokenGrid, Error> {
        self.generate_semantic_tokens_internal(
            text_tokens,
            codec_cardinality,
            seed,
            max_semantic_frames,
            Some(on_frame),
        )
    }

    fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        FishAudioTextDecoderRuntime::take_instrumentation(self)
    }
}
