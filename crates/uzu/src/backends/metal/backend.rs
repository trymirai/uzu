use std::{cell::RefCell, rc::Rc, time::Instant};

use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::Retained;

use super::{
    ForwardPassState, MTLContext, MetalArray,
    compilation_parameters::CompilationConfig,
    executable_builder::DecoderExecutables,
    forward_pass::{
        ForwardPassBuffers, ModelShape, SharedBuffers,
        encodable_with_state::EncodableWithState,
    },
    kernel::{
        KernelDataType,
        kv_cache_update::{KVCacheUpdate, KVLayerData},
        sampling::SamplingKernelEncodable,
    },
};
use crate::{
    Array, DataType,
    backends::{
        Backend, KVCache, RunResult, SamplingConfig,
        metal::forward_pass::encodable_with_state::EncodingParameters,
    },
    config::DecoderConfig,
    generator::{config::GeneratorConfig, error::GeneratorError},
    parameters::ParameterTree,
};

pub struct MetalBackend {
    context: Rc<MTLContext>,
    command_buffer: Retained<MPSCommandBuffer>,

    allow_pre_encode: bool,
    model_shape: ModelShape,

    prefix_length: usize,
    pre_encoded_run: Option<PreEncodedRun>,

    shared_buffers: Rc<RefCell<SharedBuffers>>,
    scratch_buffers: ForwardPassBuffers,
    kv_cache: KVCache<Self>,

    executables: DecoderExecutables,
    gpu_sampler: SamplingKernelEncodable,
    kv_cache_update: KVCacheUpdate,
}

pub struct MetalBackendState {
    prefix_length: usize,
    kv_cache: KVCache<MetalBackend>,
}

impl Backend for MetalBackend {
    type Context = MTLContext;
    type Array = MetalArray;
    type State = MetalBackendState;

    fn new(
        context: Rc<MTLContext>,
        generator_config: &GeneratorConfig,
        decoder_config: &DecoderConfig,
        weights: &ParameterTree<MetalBackend>,
    ) -> Result<Self, GeneratorError> {
        let command_buffer =
            MPSCommandBuffer::from_command_queue(&context.command_queue);

        let allow_pre_encode =
            !context.debug_active() && generator_config.allow_pre_encode;

        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let compilation_config = Rc::new(CompilationConfig::default());

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(
            &context,
            &decoder_config,
            &model_shape,
        )));
        shared_buffers.borrow_mut().update_data(weights);

        let generate_suffix_length = generator_config.generate_suffix_length();
        let max_prefix_length: usize = std::cmp::min(
            generator_config.context_length,
            decoder_config.context_length,
        );
        let max_suffix_length: usize = std::cmp::max(
            generator_config.prefill_step_size,
            generate_suffix_length,
        );

        let kv_cache = KVCache::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        );

        let scratch_buffers = ForwardPassBuffers::new(
            &context,
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        );

        let executables = DecoderExecutables::new(
            &context,
            decoder_config,
            weights,
            compilation_config.clone(),
        );

        let intermediate_data_type: DataType =
            decoder_config.output_norm_config.scale_precision.into();
        let kernel_data_type: KernelDataType = intermediate_data_type.into();
        let kv_cache_update =
            KVCacheUpdate::new(&context, kernel_data_type, max_prefix_length)
                .unwrap();

        let gpu_sampler = SamplingKernelEncodable::new(
            &context,
            kernel_data_type,
            max_suffix_length,
            decoder_config.vocab_size,
            generator_config.sampling_seed,
        )
        .map_err(|_| GeneratorError::UnableToCreateMetalContext)?;

        Ok(Self {
            context,
            command_buffer,
            allow_pre_encode,
            model_shape,
            prefix_length: 0,
            pre_encoded_run: None,
            shared_buffers,
            scratch_buffers,
            kv_cache,
            executables,
            gpu_sampler,
            kv_cache_update,
        })
    }

    fn context(&self) -> &Self::Context {
        &self.context
    }

    fn run(
        &mut self,
        tokens: &[u64],
        token_positions: &[usize],
        expected_amount_of_new_tokens: usize,
        sampling_config: Option<SamplingConfig>,
        warmup: bool,
    ) -> RunResult<MetalArray> {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = ForwardPassState::new(
                &self.context,
                &self.model_shape,
                &self.scratch_buffers,
                &mut self.kv_cache,
                self.shared_buffers.clone(),
                tokens,
                token_positions,
                false,
                None,
            );
            state.sampling_config = sampling_config;

            if self.pre_encoded_run
                != Some(PreEncodedRun {
                    prefix_length: self.prefix_length,
                    size: tokens.len(),
                })
            {
                // Dropping a command buffer like this does not seem to be
                // an intended usecase and generates warnings in debug mode,
                // but works fine in reality.
                self.command_buffer = MPSCommandBuffer::from_command_queue(
                    &self.context.command_queue,
                );

                self.executables.encode(
                    &mut state,
                    &self.command_buffer,
                    &EncodingParameters::new(warmup, true, false),
                );
            }

            if !warmup {
                self.gpu_sampler.encode(
                    &mut state,
                    &self.command_buffer,
                    &EncodingParameters::new(warmup, true, false),
                );
            }

            let commited_command_buffer =
                self.command_buffer.root_command_buffer().to_owned();
            self.command_buffer.commit_and_continue();

            if self.allow_pre_encode && !warmup {
                self.executables.encode(
                    &mut state,
                    &self.command_buffer,
                    &EncodingParameters::new(warmup, true, false)
                        .with_projection(expected_amount_of_new_tokens),
                );
                self.pre_encoded_run = Some(PreEncodedRun {
                    prefix_length: self.prefix_length
                        + expected_amount_of_new_tokens,
                    size: tokens.len(),
                });
            }

            commited_command_buffer.wait_until_completed();
            let run_time = run_start.elapsed().as_secs_f64();

            // TODO: sampling_output is allocated unsafely from scratch buvfers,
            // returning it breaks meomory safety guarantees.
            RunResult {
                sampling_output: state
                    .sampling_output
                    .map(|cell| cell.into_inner()),
                duration: run_time,
            }
        })
    }

    fn accept_tokens(
        &mut self,
        indices: &[usize],
    ) {
        let mut scatter_fn =
            |keys: &mut MetalArray,
             values: &mut MetalArray,
             source_indices: &[usize],
             destination_indices: &[usize]| {
                let root_cb =
                    self.command_buffer.root_command_buffer().to_owned();

                let key_buffer = unsafe { keys.mtl_buffer() }.clone();
                let value_buffer = unsafe { values.mtl_buffer() }.clone();
                let k_shape = keys.shape().to_vec();
                let v_shape = values.shape().to_vec();

                let layer_data = KVLayerData {
                    key_buffer,
                    key_shape: [k_shape[0], k_shape[1], k_shape[2]],
                    value_buffer,
                    value_shape: [v_shape[0], v_shape[1], v_shape[2]],
                };

                let _ = self.kv_cache_update.encode(
                    &[layer_data],
                    source_indices,
                    destination_indices,
                    &root_cb,
                );

                self.command_buffer.commit_and_continue();
            };

        self.kv_cache.update_after_acceptance(indices, &mut scatter_fn);

        let positions: Vec<usize> =
            (self.prefix_length..).take(indices.len()).collect();

        self.kv_cache.register_accepted_tokens(&positions);
    }

    fn clone_state(&self) -> MetalBackendState {
        MetalBackendState {
            prefix_length: self.prefix_length,
            kv_cache: self
                .kv_cache
                .clone_with_prefix_len(&self.context, self.prefix_length),
        }
    }

    fn restore_state(
        &mut self,
        state: &MetalBackendState,
    ) {
        self.prefix_length = state.prefix_length;

        for (ctx_layer, gen_layer) in
            state.kv_cache.data.iter().zip(self.kv_cache.data.iter_mut())
        {
            let copy_rows = ctx_layer.effective_prefix_length();
            if copy_rows > 0 {
                gen_layer.keys.borrow_mut().copy_slice(
                    &ctx_layer.keys.borrow(),
                    1,
                    0..copy_rows,
                    0,
                );
                gen_layer.values.borrow_mut().copy_slice(
                    &ctx_layer.values.borrow(),
                    1,
                    0..copy_rows,
                    0,
                );
            }
            gen_layer.state = ctx_layer.state.clone();
            gen_layer.prefix_token_positions =
                ctx_layer.prefix_token_positions.clone();
        }
    }

    fn reset_state(&mut self) {
        self.prefix_length = 0;
        self.kv_cache.clear();
    }

    fn prefix_length(&self) -> usize {
        return self.prefix_length;
    }
}

#[derive(PartialEq)]
struct PreEncodedRun {
    prefix_length: usize,
    size: usize,
}
