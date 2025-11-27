use std::{rc::Rc, sync::Arc};

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::layer_executables::LayerExecutables;
use crate::{
    DataType,
    backends::metal::{
        ExecutionOrchestrator, KernelDataType, MTLContext, ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, ForwardPassState, RopeType,
            encodable_with_state::{EncodableWithState, EncodingParameters},
            transformer_layer::{embed_block, readout_block},
        },
        kernel::{
            RMSNormKernelEncodable, RopeKernelEncodable,
            SamplingKernelEncodable,
        },
    },
    config::{DecoderConfig, DecoderLayerType, decoder_layer::MixerConfig},
    parameters::ParameterTree,
};

pub struct DecoderExecutables {
    pub embed: Box<dyn EncodableWithState>,
    pub layers: Box<[LayerExecutables]>,
    pub norm: Box<dyn EncodableWithState>,
    pub readout: Box<dyn EncodableWithState>,
    pub global_rope: Option<Rc<Box<dyn EncodableWithState>>>,
    pub local_rope: Option<Rc<Box<dyn EncodableWithState>>>,
}

impl DecoderExecutables {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        decoder_config: Rc<DecoderConfig>,
        decoder_weight_loader: &ParameterTree<Rc<MTLContext>>,
        compilation_config: Rc<CompilationConfig>,
    ) -> Self {
        let embed = embed_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
            decoder_weight_loader,
        );

        let readout = readout_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
            decoder_weight_loader,
        );

        let attention_data_type = Self::attention_data_type(&decoder_config);
        let norm_reference_layer = decoder_config
            .layer_configs
            .as_ref()
            .map(|configs| &configs[0])
            .unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config
        {
            MixerConfig::Attention(attention_config) => attention_config
                .qkv_projection_config
                .activation_precision()
                .into(),
            MixerConfig::Mamba(mamba_config) => {
                mamba_config.in_projection_config.activation_precision().into()
            },
        };

        let global_rope = if decoder_config.global_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| {
                Self::create_rope_block(
                    &mtl_context,
                    (*data_type).into(),
                    RopeType::Global,
                )
            })
        } else {
            None
        };

        let local_rope = if decoder_config.local_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| {
                Self::create_rope_block(
                    &mtl_context,
                    (*data_type).into(),
                    RopeType::Local,
                )
            })
        } else {
            None
        };

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes =
            model_shape.sliding_window_length_per_layer.clone();

        let layers = (0..decoder_config.num_layers)
            .map(|layer_index| {
                let layer_config = decoder_config
                    .layer_configs
                    .as_ref()
                    .map(|configs| &configs[layer_index])
                    .unwrap_or(&decoder_config.layer_config);
                let layer_type = model_shape.layer_type(layer_index);
                let rope_for_layer = match layer_type {
                    DecoderLayerType::Transformer => {
                        let mut rope_block = global_rope.clone().expect(
                            "Global rope missing for transformer layer",
                        );
                        if let (Some(_), Some(local_rope_block)) = (
                            sliding_window_sizes[layer_index],
                            local_rope.clone(),
                        ) {
                            rope_block = local_rope_block;
                        }
                        Some(rope_block)
                    },
                    DecoderLayerType::StateSpace {
                        ..
                    } => None,
                };

                let layer_loader = decoder_weight_loader
                    .subtree(&format!("layers.{}", layer_index))
                    .unwrap();

                LayerExecutables::new(
                    &mtl_context,
                    layer_config,
                    layer_type,
                    compilation_config.clone(),
                    layer_index,
                    decoder_config.model_dim,
                    decoder_config.hidden_dim,
                    decoder_config.num_heads,
                    decoder_config.head_dim,
                    decoder_config.num_groups,
                    decoder_config.attention_scale,
                    &layer_loader,
                    rope_for_layer,
                )
            })
            .collect::<Vec<_>>();

        let norm_block: Box<dyn EncodableWithState> = Box::new(
            RMSNormKernelEncodable::new(
                &mtl_context,
                norm_data_type,
                decoder_config.output_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_weight_loader.subtree("output_norm").unwrap(),
            )
            .expect("Failed to create output RMS norm kernel"),
        );

        Self {
            embed: embed,
            layers: layers.into_boxed_slice(),
            norm: norm_block,
            readout: readout,
            global_rope,
            local_rope,
        }
    }

    fn create_rope_block(
        mtl_context: &MTLContext,
        kernel_data_type: KernelDataType,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableWithState>> {
        let rotation: Box<dyn EncodableWithState> = Box::new(
            RopeKernelEncodable::new(mtl_context, kernel_data_type, rope_type)
                .expect("Failed to create RopeKernelEncodable"),
        );
        return Rc::new(rotation);
    }

    fn attention_data_type(decoder_config: &DecoderConfig) -> Option<DataType> {
        (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config.attention_config().map(|attention_config| {
                attention_config
                    .qkv_projection_config
                    .activation_precision()
                    .into()
            })
        })
    }
}

impl EncodableWithState for DecoderExecutables {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        // Detect encoder requirements based on component support
        let embed_shared = self.embed.supports_shared_encoder();
        let readout_shared = self.readout.supports_shared_encoder();

        // All quantized models: 1 encoder for entire forward pass
        // Mixed models: break at MPSGraph boundaries
        self.encode_adaptive(
            state,
            command_buffer,
            parameters,
            embed_shared,
            readout_shared,
        );
    }
}

impl DecoderExecutables {
    /// Encode embed and signal the given fence (or create new one if None)
    pub fn encode_embed_only(
        &self,
        state: &mut ForwardPassState,
        orchestrator: &ExecutionOrchestrator,
        parameters: &EncodingParameters,
        fence_to_signal: Option<&metal::Fence>,
    ) -> metal::Fence {
        let embed_shared = self.embed.supports_shared_encoder();

        // Create fresh MPS CB for embed
        let mps_command_buffer = orchestrator.new_mps_command_buffer();
        // Track MPS CB; needs_commit=true if we didn't commit (pre-encode case)
        orchestrator.set_pending_mps(&mps_command_buffer, !parameters.enable_commit);

        // Use provided fence or create new one
        let embed_fence = fence_to_signal
            .cloned()
            .unwrap_or_else(|| state.fence_registry.new_fence());

        if !embed_shared {
            // MPS embed - only commit if enable_commit is true
            self.embed.encode(state, &mps_command_buffer, parameters);
            if parameters.enable_commit {
                mps_command_buffer.commit_and_continue();
            }

            let fence_cb = orchestrator.new_command_buffer();
            fence_cb.set_label("embed-fence");
            let encoder = fence_cb.new_compute_command_encoder();
            encoder.update_fence(&embed_fence);
            encoder.end_encoding();
            orchestrator.add_pending(fence_cb);
        } else {
            let embed_cb = orchestrator.new_command_buffer();
            embed_cb.set_label("embed");
            let encoder = embed_cb.new_compute_command_encoder();
            if let Some(prev_fence) = state.fence_registry.take_previous() {
                encoder.wait_for_fence(&prev_fence);
            }
            self.embed.encode_with_shared_encoder(state, encoder, parameters);
            encoder.update_fence(&embed_fence);
            encoder.end_encoding();
            orchestrator.add_pending(embed_cb);
        }

        state.fence_registry.set_current(embed_fence.clone());
        embed_fence
    }

    /// Encode layers + norm + readout + sampler (for pre-encode, skipping embed)
    /// Returns the embed_fence that layer 0 will wait on
    pub fn encode_layers_only(
        &self,
        state: &mut ForwardPassState,
        orchestrator: &ExecutionOrchestrator,
        parameters: &EncodingParameters,
        sampler: Option<&SamplingKernelEncodable>,
    ) -> metal::Fence {
        let readout_shared = self.readout.supports_shared_encoder();

        // Create embed fence that layers will wait on
        let embed_fence = state.fence_registry.new_fence();

        // Encode layers + rest (reuse existing logic)
        self.encode_layers_and_rest(
            state,
            orchestrator,
            parameters,
            sampler,
            &embed_fence,
            readout_shared,
        );

        embed_fence
    }

    /// Parallel encoding into orchestrator - encodes all work, caller commits/waits
    pub fn encode_into_orchestrator(
        &self,
        state: &mut ForwardPassState,
        orchestrator: &ExecutionOrchestrator,
        parameters: &EncodingParameters,
        sampler: Option<&SamplingKernelEncodable>,
    ) {
        let readout_shared = self.readout.supports_shared_encoder();

        // Phase 1: Embed (always fresh)
        let embed_fence = self.encode_embed_only(state, orchestrator, parameters, None);

        // Phase 2+: Layers + norm + readout + sampler
        self.encode_layers_and_rest(
            state,
            orchestrator,
            parameters,
            sampler,
            &embed_fence,
            readout_shared,
        );
    }

    fn encode_layers_and_rest(
        &self,
        state: &mut ForwardPassState,
        orchestrator: &ExecutionOrchestrator,
        parameters: &EncodingParameters,
        sampler: Option<&SamplingKernelEncodable>,
        embed_fence: &metal::Fence,
        readout_shared: bool,
    ) {

        // Freeze state
        let required_ids = self.collect_required_buffers();
        let frozen = Arc::new(state.freeze(&required_ids));

        // Phase 2: Layers in PARALLEL
        let layer_fences: Vec<_> = (0..self.layers.len())
            .map(|_| state.fence_registry.new_fence())
            .collect();

        let layer_cbs: Vec<_> = (0..self.layers.len())
            .map(|i| {
                let cb = orchestrator.new_command_buffer();
                cb.set_label(&format!("layer-{}", i));
                cb
            })
            .collect();

        let layer_encoders: Vec<_> = self
            .layers
            .iter()
            .map(|layer| layer.create_parallel_encoder())
            .collect();

        // Parallel encode (CPU parallel, GPU sequential via fences)
        rayon::scope(|s| {
            for (i, (layer_encoder, cb)) in
                layer_encoders.iter().zip(layer_cbs.iter()).enumerate()
            {
                let frozen = &frozen;
                let fences = &layer_fences;
                let params = parameters.clone();
                let ef = &embed_fence;

                s.spawn(move |_| {
                    let encoder = cb.new_compute_command_encoder();
                    if i == 0 {
                        encoder.wait_for_fence(ef);
                    } else {
                        encoder.wait_for_fence(&fences[i - 1]);
                    }
                    layer_encoder.encode(encoder, frozen, &params);
                    encoder.update_fence(&fences[i]);
                    encoder.end_encoding();
                });
            }
        });

        // Add layer CBs to orchestrator (order matters for commit)
        for cb in layer_cbs {
            orchestrator.add_pending(cb);
        }

        // Phase 3: Norm + Readout + Sampler (if not prefilling)
        if !state.is_prefilling() {
            let final_cb = orchestrator.new_command_buffer();
            final_cb.set_label("norm-readout-sample");
            let encoder = final_cb.new_compute_command_encoder();

            // Wait on last layer
            if let Some(last_fence) = layer_fences.last() {
                encoder.wait_for_fence(last_fence);
            }

            // Norm
            self.norm.encode_with_shared_encoder(state, encoder, parameters);

            // Readout (shared encoder path)
            if readout_shared {
                self.readout
                    .encode_with_shared_encoder(state, encoder, parameters);
            } else {
                // MPS readout - signal fence, end encoder, commit CB, then MPS
                let readout_fence = state.fence_registry.new_fence();
                encoder.update_fence(&readout_fence);
                encoder.end_encoding();
                orchestrator.add_pending(final_cb);

                // Fresh MPS CB for readout - only commit if enable_commit
                let readout_mps = orchestrator.new_mps_command_buffer();
                self.readout.encode(state, &readout_mps, parameters);
                if parameters.enable_commit {
                    readout_mps.commit_and_continue();
                }
                orchestrator.set_pending_mps(&readout_mps, !parameters.enable_commit);

                state.fence_registry.set_current(readout_fence);

                // Sampler needs separate CB after MPS readout
                if let Some(sampler) = sampler {
                    if !parameters.warmup {
                        let sample_cb = orchestrator.new_command_buffer();
                        sample_cb.set_label("sample");
                        let sample_encoder =
                            sample_cb.new_compute_command_encoder();
                        if let Some(prev) = state.fence_registry.take_previous()
                        {
                            sample_encoder.wait_for_fence(&prev);
                        }
                        sampler
                            .encode_with_shared_encoder(state, sample_encoder);
                        let sample_fence = state.fence_registry.new_fence();
                        sample_encoder.update_fence(&sample_fence);
                        sample_encoder.end_encoding();
                        orchestrator.add_pending(sample_cb);
                        state.fence_registry.set_current(sample_fence);
                    }
                }
                return;
            }

            // Sampler (on same encoder as norm+readout)
            if let Some(sampler) = sampler {
                if !parameters.warmup {
                    sampler.encode_with_shared_encoder(state, encoder);
                }
            }

            // Final fence
            let final_fence = state.fence_registry.new_fence();
            encoder.update_fence(&final_fence);
            encoder.end_encoding();
            orchestrator.add_pending(final_cb);
            state.fence_registry.set_current(final_fence);
        }
        // Caller calls orchestrator.commit() and orchestrator.wait()
    }
}

impl DecoderExecutables {
    fn encode_adaptive(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
        embed_shared: bool,
        readout_shared: bool,
    ) {
        let cmd = command_buffer.root_command_buffer();

        // Phase 1: Embed
        if !embed_shared {
            // MPSGraph embed creates its own encoders
            self.embed.encode(state, command_buffer, parameters);
        }

        // Phase 2: Main encoder (embed if shared + layers + norm + readout if shared)
        let encoder = cmd.new_compute_command_encoder();

        // Wait on previous fence
        if let Some(prev_fence) = state.fence_registry.take_previous() {
            encoder.wait_for_fence(&prev_fence);
        }

        // Embed (if shared encoder)
        if embed_shared {
            self.embed.encode_with_shared_encoder(state, encoder, parameters);
        }

        // All layers
        for layer in self.layers.iter() {
            layer.encode_with_shared_encoder(state, encoder, parameters);
        }

        // Norm + Readout (if not prefilling)
        if !state.is_prefilling() {
            self.norm.encode_with_shared_encoder(state, encoder, parameters);

            if let Some(traces) = state.traces.clone() {
                state.copy_array(
                    ArrayId::Main,
                    traces.borrow().output_norm.clone(),
                );
            }

            if readout_shared {
                self.readout
                    .encode_with_shared_encoder(state, encoder, parameters);

                if let Some(traces) = state.traces.clone() {
                    state.copy_array(
                        ArrayId::Logits,
                        traces.borrow().logits.clone(),
                    );
                }
            }
        }

        // Signal fence and end encoder
        let fence = state.fence_registry.new_fence();
        encoder.update_fence(&fence);
        encoder.end_encoding();
        state.fence_registry.set_current(fence);

        // Phase 3: Readout (if MPSGraph)
        if !state.is_prefilling() && !readout_shared {
            self.readout.encode(state, command_buffer, parameters);

            if let Some(traces) = state.traces.clone() {
                state.copy_array(
                    ArrayId::Logits,
                    traces.borrow().logits.clone(),
                );
            }
        }
    }

    /// Collect all ArrayIds needed by all layers for parallel encoding
    pub fn collect_required_buffers(&self) -> Vec<ArrayId> {
        let mut ids = Vec::new();

        // Core buffers used by all layers
        ids.push(ArrayId::Main);
        ids.push(ArrayId::Shortcut);
        ids.push(ArrayId::QKV);
        ids.push(ArrayId::AttentionOutput);
        ids.push(ArrayId::MlpFusedUp);
        ids.push(ArrayId::MlpHidden);
        ids.push(ArrayId::RotatedQueries);
        ids.push(ArrayId::RotatedKeys);
        ids.push(ArrayId::AttentionPartials);
        ids.push(ArrayId::AttentionSums);
        ids.push(ArrayId::AttentionMaxs);

        // Per-layer KV cache
        for i in 0..self.layers.len() {
            ids.push(ArrayId::Keys(i));
            ids.push(ArrayId::Values(i));
        }

        // SSM buffers (if any layers use SSM)
        ids.push(ArrayId::SsmInProj);
        ids.push(ArrayId::SsmConvPadded);
        for i in 0..self.layers.len() {
            ids.push(ArrayId::SsmConvState(i));
            ids.push(ArrayId::SsmState(i));
            ids.push(ArrayId::SsmPacked(i));
            ids.push(ArrayId::SsmX(i));
            ids.push(ArrayId::SsmB(i));
            ids.push(ArrayId::SsmC(i));
            ids.push(ArrayId::SsmDt(i));
            ids.push(ArrayId::SsmZ(i));
        }

        ids
    }

    /// Test method: encode layers using frozen state + encode_parallel
    /// This validates the FrozenState path works correctly
    pub fn encode_with_frozen_state_test(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let embed_shared = self.embed.supports_shared_encoder();
        let readout_shared = self.readout.supports_shared_encoder();

        // Phase 1: Embed (if not shared, encode separately)
        if !embed_shared {
            self.embed.encode(state, command_buffer, parameters);
        }

        // Collect required buffers for all layers
        let required_ids = self.collect_required_buffers();

        // Create frozen state snapshot
        let frozen = state.freeze(&required_ids);

        // Phase 2: Main compute pass with frozen state
        let cmd = command_buffer.root_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        // Wait on previous fence
        if let Some(prev_fence) = state.fence_registry.take_previous() {
            encoder.wait_for_fence(&prev_fence);
        }

        // Embed (if shared)
        if embed_shared {
            self.embed.encode_with_shared_encoder(state, encoder, parameters);
        }

        // Encode all layers using encode_parallel
        for layer in self.layers.iter() {
            if layer.supports_parallel_encode() {
                layer.encode_parallel_direct(encoder, &frozen, parameters);
            } else {
                layer.encode_with_shared_encoder(state, encoder, parameters);
            }
        }

        // Norm + Readout (if not prefilling)
        if !state.is_prefilling() {
            self.norm.encode_with_shared_encoder(state, encoder, parameters);

            if readout_shared {
                self.readout
                    .encode_with_shared_encoder(state, encoder, parameters);
            }
        }

        // Signal fence and end encoder
        let fence = state.fence_registry.new_fence();
        encoder.update_fence(&fence);
        encoder.end_encoding();
        state.fence_registry.set_current(fence);

        // Phase 3: Readout (if MPSGraph)
        if !state.is_prefilling() && !readout_shared {
            self.readout.encode(state, command_buffer, parameters);
        }

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            cmd.wait_until_completed();
        }
    }
}
