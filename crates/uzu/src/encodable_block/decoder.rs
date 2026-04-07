//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::{
        Backend, Encoder,
        kernel::{Kernels as KernelsTrait, RMSNormKernel, TensorAddScaleKernel},
    },
    config::{
        DecoderConfig, DecoderLayerType, EmbeddingConfig, EmbeddingConfigCommon, MixerConfig, NormalizationConfig,
        UpcastMode,
    },
    encodable_block::{
        Embedding, EncodingParameters, LayerExecutables, Linear, RMSNorm, Rope, embedding::EmbeddingError,
    },
    forward_pass::{
        model_shape::ModelShape,
        state::{ArrayId, ForwardPassState, RopeType},
    },
    parameters::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
}

/// PLE model-level weights loaded from the weight tree.
struct PleModelWeights<B: Backend> {
    embed: Option<Embedding<B>>,
    projection: Option<Box<dyn Linear<B>>>,
    norm_kernel: Option<<B::Kernels as KernelsTrait>::RMSNormKernel>,
    norm_scales: Option<Rc<RefCell<B::Buffer>>>,
    norm_config: Option<NormalizationConfig>,
    scale_kernel: Option<<B::Kernels as KernelsTrait>::TensorAddScaleKernel>,
    zero_bias: Option<Array<B>>,
}

impl<B: Backend> Default for PleModelWeights<B> {
    fn default() -> Self {
        Self {
            embed: None,
            projection: None,
            norm_kernel: None,
            norm_scales: None,
            norm_config: None,
            scale_kernel: None,
            zero_bias: None,
        }
    }
}

/// Full decoder executable with all layers and components.
pub struct Decoder<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Box<[LayerExecutables<B>]>,
    pub norm: RMSNorm<B>,
    /// PLE: per-layer embedding lookup table [vocab_size, ple_total_dim] (embedding index lookup)
    pub ple_embed: Option<Embedding<B>>,
    /// PLE: model projection [model_dim, ple_dim] (projects main embeddings into PLE space)
    pub ple_projection: Option<Box<dyn Linear<B>>>,
    /// PLE: RMSNorm kernel for normalizing projected embeddings (norm dim = ple_dim, NOT ple_total_dim)
    ple_norm_kernel: Option<<B::Kernels as KernelsTrait>::RMSNormKernel>,
    /// PLE: RMSNorm scales buffer [ple_dim] for per-layer-chunk normalization
    ple_norm_scales: Option<Rc<RefCell<B::Buffer>>>,
    /// PLE: RMSNorm config (epsilon, scale_offset, upcast_mode)
    ple_norm_config: Option<NormalizationConfig>,
    /// PLE: TensorAddScale kernel for scaling and combining PLE buffers
    ple_scale_kernel: Option<<B::Kernels as KernelsTrait>::TensorAddScaleKernel>,
    /// PLE: zero-valued bias buffer [ple_total_dim] used for scalar-only multiply via TensorAddScale
    ple_zero_bias: Option<Array<B>>,
    /// PLE: scale applied to the projected embeddings before normalization
    ple_projection_scale: f32,
    /// PLE: scale applied when combining projection + embedding
    ple_combination_scale: f32,
    /// PLE: number of layers (needed for RMSNorm batch dimension)
    ple_num_layers: usize,
    /// PLE: per-layer embedding dimension (normalization dimension)
    ple_dim: usize,
}

impl<B: Backend> Decoder<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
    ) -> Self {
        let embedding_weight_loader = root_weight_loader.subtree("embedding").expect("Failed to get embedding subtree");

        let embed = Embedding::new(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
        )
        .expect("Failed to create embedding");

        let (layers, norm) =
            Self::build_transformer_layers_and_norm(context, decoder_config, root_weight_loader, "transformer");

        let ple = Self::load_ple_model_weights(context, decoder_config, root_weight_loader);

        Self {
            embed,
            layers,
            norm,
            ple_embed: ple.embed,
            ple_projection: ple.projection,
            ple_norm_kernel: ple.norm_kernel,
            ple_norm_scales: ple.norm_scales,
            ple_norm_config: ple.norm_config,
            ple_scale_kernel: ple.scale_kernel,
            ple_zero_bias: ple.zero_bias,
            ple_projection_scale: decoder_config.ple_projection_scale.unwrap_or(1.0),
            ple_combination_scale: decoder_config.ple_combination_scale.unwrap_or(1.0),
            ple_num_layers: decoder_config.num_layers,
            ple_dim: decoder_config.ple_dim.unwrap_or(0),
        }
    }

    /// Used by models whose token lookup weights and logits readout weights
    #[cfg(all(feature = "audio-runtime", metal_backend))]
    pub fn new_with_embedding_and_readout_subtrees(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
    ) -> Self {
        let embedding_weight_loader =
            root_weight_loader.subtree(embedding_subtree).expect("Failed to get embedding subtree");
        let readout_weight_loader = root_weight_loader.subtree(readout_subtree).expect("Failed to get readout subtree");

        let embed = Embedding::new_with_lookup_and_readout_trees(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
            &readout_weight_loader,
        )
        .expect("Failed to create embedding");

        let (layers, norm) =
            Self::build_transformer_layers_and_norm(context, decoder_config, root_weight_loader, transformer_subtree);

        let ple = Self::load_ple_model_weights(context, decoder_config, root_weight_loader);

        Self {
            embed,
            layers,
            norm,
            ple_embed: ple.embed,
            ple_projection: ple.projection,
            ple_norm_kernel: ple.norm_kernel,
            ple_norm_scales: ple.norm_scales,
            ple_norm_config: ple.norm_config,
            ple_scale_kernel: ple.scale_kernel,
            ple_zero_bias: ple.zero_bias,
            ple_projection_scale: decoder_config.ple_projection_scale.unwrap_or(1.0),
            ple_combination_scale: decoder_config.ple_combination_scale.unwrap_or(1.0),
            ple_num_layers: decoder_config.num_layers,
            ple_dim: decoder_config.ple_dim.unwrap_or(0),
        }
    }

    pub(crate) fn build_transformer_layers_and_norm(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
        transformer_subtree: &str,
    ) -> (Box<[LayerExecutables<B>]>, RMSNorm<B>) {
        let decoder_weight_loader =
            root_weight_loader.subtree(transformer_subtree).expect("transformer subtree not found");

        let attention_data_type = Self::attention_data_type(&decoder_config);
        let norm_reference_layer =
            decoder_config.layer_configs.as_ref().map(|configs| &configs[0]).unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config {
            MixerConfig::Attention(attention_config) => {
                attention_config.qkv_projection_config.activation_precision().into()
            },
            MixerConfig::Mamba(mamba_config) => mamba_config.in_projection_config.activation_precision().into(),
            MixerConfig::ShortConv(short_conv_config) => {
                short_conv_config.in_projection_config.activation_precision().into()
            },
            MixerConfig::DeltaNet(config) => config.in_proj_config.activation_precision().into(),
        };

        let global_rope = if decoder_config.global_rope_config.is_some() {
            attention_data_type
                .as_ref()
                .map(|data_type| Self::create_rope_block(&context, *data_type, RopeType::Global))
        } else {
            None
        };

        let local_rope = if decoder_config.local_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| Self::create_rope_block(&context, *data_type, RopeType::Local))
        } else {
            None
        };

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes = model_shape.sliding_window_length_per_layer.clone();

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
                        if let Some(_) = sliding_window_sizes[layer_index]
                            && let Some(local_rope_block) = local_rope.clone()
                        {
                            Some(local_rope_block)
                        } else {
                            Some(global_rope.clone().expect("Global rope missing for transformer layer"))
                        }
                    },
                    DecoderLayerType::StateSpace {
                        ..
                    } => None,
                    DecoderLayerType::ShortConv {
                        ..
                    } => None,
                    DecoderLayerType::DeltaNet {
                        ..
                    } => None,
                };

                let layer_loader = decoder_weight_loader.subtree(&format!("layers.{}", layer_index)).unwrap();

                LayerExecutables::new(
                    context,
                    layer_config,
                    layer_type,
                    layer_index,
                    decoder_config.model_dim,
                    decoder_config
                        .hidden_dims
                        .as_ref()
                        .map(|dims| dims[layer_index])
                        .unwrap_or(decoder_config.hidden_dim),
                    decoder_config.num_heads,
                    decoder_config.head_dim,
                    decoder_config.num_groups,
                    decoder_config.attention_scale,
                    &layer_loader,
                    rope_for_layer,
                    decoder_config.ple_dim,
                    decoder_config.ple_linear_config.as_ref(),
                    decoder_config.ple_norm_config.as_ref(),
                    decoder_config.num_layers,
                    decoder_config
                        .kv_shared_layer_sources
                        .as_ref()
                        .and_then(|sources| sources.get(layer_index).copied().flatten())
                        .is_some(),
                )
            })
            .collect::<Vec<_>>();

        let norm_block = RMSNorm::new(
            context,
            norm_data_type,
            decoder_config.output_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &decoder_weight_loader.subtree("output_norm").unwrap(),
            Some(ArrayId::Shortcut),
            true,
        )
        .map(RMSNorm::with_sampling_range)
        .expect("Failed to create output RMS norm kernel");

        (layers.into_boxed_slice(), norm_block)
    }

    fn load_ple_model_weights(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
    ) -> PleModelWeights<B> {
        let ple_dim = match decoder_config.ple_dim {
            Some(dim) if dim > 0 => dim,
            _ => return PleModelWeights::default(),
        };
        let ple_linear_config =
            decoder_config.ple_linear_config.as_ref().expect("ple_linear_config required when ple_dim > 0");
        let ple_norm_config =
            decoder_config.ple_norm_config.as_ref().expect("ple_norm_config required when ple_dim > 0");

        let ple_loader = root_weight_loader.subtree("ple").expect("PLE subtree not found");
        let ple_data_type: DataType = ple_linear_config.activation_precision().into();

        let ple_total_dim = decoder_config.num_layers * ple_dim;

        // embed_tokens_per_layer: [vocab_size, ple_total_dim] — embedding index lookup by token ID
        let ple_embed_config = EmbeddingConfig::Untied {
            common: EmbeddingConfigCommon {
                input_scale: Some(decoder_config.ple_embed_scale.unwrap_or(1.0)),
                logit_soft_cap: None,
            },
            precision: ple_linear_config.activation_precision(),
        };

        let ple_embed = Embedding::new_lookup_only(
            context,
            decoder_config.vocab_size as u32,
            ple_total_dim as u32,
            &ple_embed_config,
            &ple_loader.subtree("embed_tokens_per_layer").unwrap(),
        )
        .expect("Failed to create PLE embed_tokens_per_layer");

        // per_layer_model_projection: projects model_dim → ple_total_dim
        let ple_projection = <dyn Linear<B>>::new(
            ple_linear_config,
            false,
            decoder_config.model_dim,
            [ple_total_dim],
            context,
            &ple_loader.subtree("per_layer_model_projection").unwrap(),
            ArrayId::Main,
            ArrayId::PleProjection,
        )
        .expect("Failed to create PLE per_layer_model_projection");

        // per_layer_projection_norm: RMSNorm kernel + scales (norm over ple_dim per layer-chunk)
        // We use the kernel directly instead of the RMSNorm encodable because:
        // - The scale weight has [ple_dim=256] elements, not [ple_total_dim=8960]
        // - We need batch_len = suffix_length * num_layers, not just suffix_length
        // - The norm operates on [suffix_length * num_layers, ple_dim] reshaped view
        let norm_scales = ple_loader
            .subtree("per_layer_projection_norm")
            .unwrap()
            .leaf_array("scales")
            .expect("Failed to load PLE norm scales");
        let accumulation_data_type: DataType = ple_norm_config.accumulation_precision.into();
        let scale_data_type: DataType = ple_norm_config.scale_precision.into();
        let (input_type, scales_type, output_type) = match ple_norm_config.upcast_mode {
            UpcastMode::OnlyNormalization => (ple_data_type, scale_data_type, scale_data_type),
            UpcastMode::FullLayer => (ple_data_type, scale_data_type, scale_data_type),
        };
        let ple_norm_kernel = <B::Kernels as KernelsTrait>::RMSNormKernel::new(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            true,
            false,
            false,
        )
        .expect("Failed to create PLE RMSNorm kernel");

        // TensorAddScale kernel used for scaling PleProjection and combining PleProjection + PleEmbeddings
        let ple_scale_kernel = <B::Kernels as KernelsTrait>::TensorAddScaleKernel::new(context, ple_data_type)
            .expect("Failed to create TensorAddScale kernel for PLE");

        // Zero-valued bias buffer for scalar-only multiply via TensorAddScale (input + 0) * scale
        let ple_zero_bias = context.create_array_zeros(&[ple_total_dim], ple_data_type, "ple_zero_bias");

        PleModelWeights {
            embed: Some(ple_embed),
            projection: Some(ple_projection),
            norm_kernel: Some(ple_norm_kernel),
            norm_scales: Some(norm_scales.buffer()),
            norm_config: Some(ple_norm_config.clone()),
            scale_kernel: Some(ple_scale_kernel),
            zero_bias: Some(ple_zero_bias),
        }
    }

    fn create_rope_block(
        context: &B::Context,
        data_type: DataType,
        rope_type: RopeType,
    ) -> Rc<Rope<B>> {
        Rc::new(Rope::<B>::new(context, data_type, rope_type).expect("Failed to create Rope"))
    }

    fn attention_data_type(decoder_config: &DecoderConfig) -> Option<DataType> {
        (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config
                .attention_config()
                .map(|attention_config| attention_config.qkv_projection_config.activation_precision().into())
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<(), DecoderError<B>> {
        self.embed.encode_lookup(state, encoder)?;

        // PLE model-level computation: embedding lookup + projection + normalization + combine
        if let (Some(ple_embed), Some(ple_projection)) = (&self.ple_embed, &self.ple_projection) {
            // Step 1: PLE embedding lookup → PleEmbeddings [seq, ple_total_dim]
            // input_scale (ple_embed_scale) is applied automatically by the embedding lookup kernel.
            ple_embed.encode_lookup_to(state, encoder, ArrayId::PleEmbeddings)?;

            // Step 2: Project main embeddings → PleProjection [seq, ple_total_dim]
            ple_projection.encode(state, encoder).map_err(DecoderError::BackendError)?;

            // Step 3: Scale PleProjection by ple_projection_scale (in-place via TensorAddScale with zero bias)
            {
                let ple_proj = state.array(ArrayId::PleProjection);
                let length = ple_proj.num_elements();
                let zero_bias = self.ple_zero_bias.as_ref().unwrap();
                let kernel = self.ple_scale_kernel.as_ref().unwrap();

                let proj_buffer_rc = ple_proj.buffer();
                let mut proj_buffer = proj_buffer_rc.borrow_mut();
                // TensorAddScale is element-wise, so in-place read/write aliasing is valid.
                let proj_input: &B::Buffer = unsafe { &*(&*proj_buffer as *const B::Buffer) };

                let num_cols = zero_bias.num_elements() as u32;
                kernel.encode(
                    (proj_input, ple_proj.offset()),
                    &*zero_bias.buffer().borrow(),
                    (&mut *proj_buffer, ple_proj.offset()),
                    num_cols,
                    length as u32,
                    self.ple_projection_scale,
                    encoder,
                );
            }

            // Step 4: RMSNorm on PleProjection (in-place, BEFORE adding embeddings)
            // HF: per_layer_projection = per_layer_projection_norm(per_layer_projection)
            // The norm weight has [ple_dim=256] elements. The buffer is [suffix_length, ple_total_dim=8960].
            // We treat it as [suffix_length * num_layers, ple_dim] so norm operates over 256 elements
            // per layer-chunk, matching the HF reshape-then-norm pattern.
            {
                let ple_proj = state.array(ArrayId::PleProjection);
                let ple_norm_kernel = self.ple_norm_kernel.as_ref().unwrap();
                let ple_norm_scales = self.ple_norm_scales.as_ref().unwrap();
                let ple_norm_config = self.ple_norm_config.as_ref().unwrap();

                let active_rows = state.active_row_count();
                let batch_len = (active_rows * self.ple_num_layers) as u32;
                let element_count = self.ple_dim as u32;

                ple_norm_kernel.encode(
                    None::<(&B::Buffer, usize)>,
                    ple_norm_scales.borrow().deref(),
                    (ple_proj.buffer().borrow_mut().deref_mut(), ple_proj.offset()),
                    None::<(&mut B::Buffer, usize)>,
                    batch_len,
                    element_count,
                    ple_norm_config.epsilon,
                    ple_norm_config.scale_offset.unwrap_or(0.0),
                    ple_norm_config.upcast_mode == UpcastMode::FullLayer,
                    encoder,
                );
            }

            // Step 5: Combine (add only): PlePerLayerInputs = PleProjection (now normed) + PleEmbeddings
            // HF: return (per_layer_projection + per_layer_inputs) * input_scale
            {
                let ple_proj = state.array(ArrayId::PleProjection);
                let ple_embed_arr = state.array(ArrayId::PleEmbeddings);
                let ple_combined = state.array(ArrayId::PlePerLayerInputs);
                let length = ple_proj.num_elements();
                let kernel = self.ple_scale_kernel.as_ref().unwrap();

                debug_assert_eq!(ple_embed_arr.offset(), 0, "PLE combine assumes zero offset on embeddings bias");
                kernel.encode(
                    (&*ple_proj.buffer().borrow(), ple_proj.offset()),
                    &*ple_embed_arr.buffer().borrow(),
                    (&mut *ple_combined.buffer().borrow_mut(), ple_combined.offset()),
                    length as u32,
                    length as u32,
                    1.0, // no scale yet — just add
                    encoder,
                );
            }

            // Step 6: Scale PlePerLayerInputs by ple_combination_scale (0.7071)
            {
                let ple_combined = state.array(ArrayId::PlePerLayerInputs);
                let length = ple_combined.num_elements();
                let zero_bias = self.ple_zero_bias.as_ref().unwrap();
                let kernel = self.ple_scale_kernel.as_ref().unwrap();

                let combined_buffer_rc = ple_combined.buffer();
                let mut combined_buffer = combined_buffer_rc.borrow_mut();
                let combined_input: &B::Buffer = unsafe { &*(&*combined_buffer as *const B::Buffer) };

                let num_cols = zero_bias.num_elements() as u32;
                kernel.encode(
                    (combined_input, ple_combined.offset()),
                    &*zero_bias.buffer().borrow(),
                    (&mut *combined_buffer, ple_combined.offset()),
                    num_cols,
                    length as u32,
                    self.ple_combination_scale,
                    encoder,
                );
            }
        }

        for layer in self.layers.iter() {
            layer.encode(state, parameters, encoder).map_err(DecoderError::BackendError)?;
        }

        if state.is_prefilling() {
            return Ok(());
        }

        self.norm.encode(state, encoder).map_err(DecoderError::BackendError)?;
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(encoder, ArrayId::Main, traces.borrow().output_norm.clone());
        }

        self.embed.encode_readout(state, encoder)?;
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(encoder, ArrayId::Logits, traces.borrow().logits.clone());
        }
        Ok(())
    }
}
