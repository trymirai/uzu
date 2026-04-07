//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use crate::{
    DataType,
    array::ArrayContextExt,
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

struct PleModelWeights<B: Backend> {
    embed: Embedding<B>,
    projection: Box<dyn Linear<B>>,
    norm_kernel: <B::Kernels as KernelsTrait>::RMSNormKernel,
    norm_scales: Rc<RefCell<B::Buffer>>,
    norm_config: NormalizationConfig,
    scale_kernel: <B::Kernels as KernelsTrait>::TensorAddScaleKernel,
    zero_bias: B::Buffer,
    zero_bias_len: usize,
    projection_scale: f32,
    combination_scale: f32,
    num_layers: usize,
    dim: usize,
}

pub struct Decoder<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Box<[LayerExecutables<B>]>,
    pub norm: RMSNorm<B>,
    ple: Option<PleModelWeights<B>>,
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

        Self {
            embed,
            layers,
            norm,
            ple: Self::load_ple_model_weights(context, decoder_config, root_weight_loader),
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

        Self {
            embed,
            layers,
            norm,
            ple: Self::load_ple_model_weights(context, decoder_config, root_weight_loader),
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

                // When the previous layer did an explicit residual add (PLE or layer_scalar),
                // this layer's pre_attention_norm must NOT fuse a residual add — the
                // previous layer's output already includes the full residual sum.
                let previous_layer_did_explicit_residual = if layer_index > 0 {
                    let prev_has_ple = decoder_config.ple_dim.map_or(false, |d| d > 0);
                    let prev_has_scalar = decoder_config
                        .layer_configs
                        .as_ref()
                        .and_then(|lcs| lcs.get(layer_index - 1))
                        .map_or(false, |l| l.has_layer_scalar);
                    prev_has_ple || prev_has_scalar
                } else {
                    false
                };

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
                    previous_layer_did_explicit_residual,
                )
            })
            .collect::<Vec<_>>();

        let last_layer_has_ple = decoder_config.ple_dim.map_or(false, |d| d > 0);
        let last_layer_has_scalar =
            decoder_config.layer_configs.as_ref().and_then(|lcs| lcs.last()).map_or(false, |l| l.has_layer_scalar);
        let output_norm_residual_add = !(last_layer_has_ple || last_layer_has_scalar);
        let norm_block = RMSNorm::new(
            context,
            norm_data_type,
            decoder_config.output_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &decoder_weight_loader.subtree("output_norm").unwrap(),
            Some(ArrayId::Shortcut),
            output_norm_residual_add,
        )
        .map(RMSNorm::with_sampling_range)
        .expect("Failed to create output RMS norm kernel");

        (layers.into_boxed_slice(), norm_block)
    }

    fn load_ple_model_weights(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
    ) -> Option<PleModelWeights<B>> {
        let ple_dim = match decoder_config.ple_dim {
            Some(dim) if dim > 0 => dim,
            _ => return None,
        };
        let ple_linear_config =
            decoder_config.ple_linear_config.as_ref().expect("ple_linear_config required when ple_dim > 0");
        let ple_norm_config =
            decoder_config.ple_norm_config.as_ref().expect("ple_norm_config required when ple_dim > 0");

        let ple_loader = root_weight_loader.subtree("ple").expect("PLE subtree not found");
        let ple_data_type: DataType = ple_linear_config.activation_precision().into();

        let ple_total_dim = decoder_config.num_layers * ple_dim;

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

        // We use the kernel directly instead of the RMSNorm encodable because the scale
        // weight has [ple_dim] elements and we need batch_len = suffix_length * num_layers,
        // operating on a [suffix_length * num_layers, ple_dim] reshaped view.
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

        let ple_scale_kernel = <B::Kernels as KernelsTrait>::TensorAddScaleKernel::new(context, ple_data_type)
            .expect("Failed to create TensorAddScale kernel for PLE");

        let ple_zero_bias_arr = context.create_array_zeros(&[ple_total_dim], ple_data_type, "ple_zero_bias");
        let ple_zero_bias_len = ple_zero_bias_arr.num_elements();
        let ple_zero_bias_rc = ple_zero_bias_arr.buffer();
        drop(ple_zero_bias_arr);
        let ple_zero_bias = Rc::try_unwrap(ple_zero_bias_rc).expect("unique owner").into_inner();

        Some(PleModelWeights {
            embed: ple_embed,
            projection: ple_projection,
            norm_kernel: ple_norm_kernel,
            norm_scales: norm_scales.buffer(),
            norm_config: ple_norm_config.clone(),
            scale_kernel: ple_scale_kernel,
            zero_bias: ple_zero_bias,
            zero_bias_len: ple_zero_bias_len,
            projection_scale: decoder_config.ple_projection_scale.unwrap_or(1.0),
            combination_scale: decoder_config.ple_combination_scale.unwrap_or(1.0),
            num_layers: decoder_config.num_layers,
            dim: ple_dim,
        })
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

        if let Some(ref ple) = self.ple {
            ple.embed.encode_lookup_to(state, encoder, ArrayId::PleEmbeddings)?;

            ple.projection.encode(state, encoder).map_err(DecoderError::BackendError)?;

            // Scale PleProjection by projection_scale (in-place via TensorAddScale with zero bias)
            {
                let ple_proj = state.array(ArrayId::PleProjection);
                let length = ple_proj.num_elements();

                let proj_buffer_rc = ple_proj.buffer();
                let mut proj_buffer = proj_buffer_rc.borrow_mut();
                // TensorAddScale is element-wise, so in-place read/write aliasing is valid.
                let proj_input: &B::Buffer = unsafe { &*(&*proj_buffer as *const B::Buffer) };

                let num_cols = ple.zero_bias_len as u32;
                ple.scale_kernel.encode(
                    (proj_input, ple_proj.offset()),
                    &ple.zero_bias,
                    (&mut *proj_buffer, ple_proj.offset()),
                    num_cols,
                    length as u32,
                    ple.projection_scale,
                    encoder,
                );
            }

            // RMSNorm on PleProjection. The norm weight has [ple_dim] elements but the buffer
            // is [suffix_length, ple_total_dim]. We treat it as [suffix_length * num_layers, ple_dim]
            // so norm operates per layer-chunk, matching the HF reshape-then-norm pattern.
            {
                let ple_proj = state.array(ArrayId::PleProjection);

                let active_rows = state.active_row_count();
                let batch_len = (active_rows * ple.num_layers) as u32;
                let element_count = ple.dim as u32;

                ple.norm_kernel.encode(
                    None::<(&B::Buffer, usize)>,
                    ple.norm_scales.borrow().deref(),
                    (ple_proj.buffer().borrow_mut().deref_mut(), ple_proj.offset()),
                    None::<(&mut B::Buffer, usize)>,
                    batch_len,
                    element_count,
                    ple.norm_config.epsilon,
                    ple.norm_config.scale_offset.unwrap_or(0.0),
                    ple.norm_config.upcast_mode == UpcastMode::FullLayer,
                    encoder,
                );
            }

            {
                let ple_proj = state.array(ArrayId::PleProjection);
                let ple_embed_arr = state.array(ArrayId::PleEmbeddings);
                let ple_combined = state.array(ArrayId::PlePerLayerInputs);
                let length = ple_proj.num_elements();

                debug_assert_eq!(ple_embed_arr.offset(), 0, "PLE combine assumes zero offset on embeddings bias");
                ple.scale_kernel.encode(
                    (&*ple_proj.buffer().borrow(), ple_proj.offset()),
                    &*ple_embed_arr.buffer().borrow(),
                    (&mut *ple_combined.buffer().borrow_mut(), ple_combined.offset()),
                    length as u32,
                    length as u32,
                    1.0, // no scale yet — just add
                    encoder,
                );
            }

            {
                let ple_combined = state.array(ArrayId::PlePerLayerInputs);
                let length = ple_combined.num_elements();

                let combined_buffer_rc = ple_combined.buffer();
                let mut combined_buffer = combined_buffer_rc.borrow_mut();
                let combined_input: &B::Buffer = unsafe { &*(&*combined_buffer as *const B::Buffer) };

                let num_cols = ple.zero_bias_len as u32;
                ple.scale_kernel.encode(
                    (combined_input, ple_combined.offset()),
                    &ple.zero_bias,
                    (&mut *combined_buffer, ple_combined.offset()),
                    num_cols,
                    length as u32,
                    ple.combination_scale,
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
