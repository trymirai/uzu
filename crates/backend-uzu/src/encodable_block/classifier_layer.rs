use std::rc::Rc;

use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder},
    config::TransformerLayerConfig,
    encodable_block::{
        Attention, AttentionArguments, EncodingParameters, LayerArguments, Linear, Mlp, Normalization, QKNorm, Rope,
        TensorAddSwap,
    },
    forward_pass::state::RopeType,
    parameters::ParameterTree,
};

pub struct ClassifierLayer<B: Backend> {
    pre_attention_norm: Option<Normalization<B>>,
    qkv_projection: Box<dyn Linear<B>>,
    qk_norm: Option<QKNorm<B>>,
    rope: Rc<Rope<B>>,
    attention: Attention<B>,
    out_projection: Box<dyn Linear<B>>,
    post_attention_norm: Option<Normalization<B>>,
    mixer_residual_add: TensorAddSwap<B>,
    pre_mlp_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
    post_mlp_norm: Option<Normalization<B>>,
    mlp_residual_add: TensorAddSwap<B>,
    model_dim: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
}

impl<B: Backend> ClassifierLayer<B> {
    pub fn new(
        context: Rc<B::Context>,
        layer_config: &TransformerLayerConfig,
        layer_index: usize,
        model_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        attention_scale: Option<f32>,
        layer_loader: &ParameterTree<B::Context>,
        rope: Rc<Rope<B>>,
    ) -> Self {
        let ctx = context.as_ref(); // Reference for functions expecting &B::Context
        let attention_config = layer_config.mixer_config.as_attention().expect("Classifier layers must use attention");
        let intermediate_data_type: DataType = attention_config.qkv_projection_config.activation_precision().into();

        let pre_attention_norm = if let Some(norm_config) = &layer_config.pre_attention_norm_config {
            if layer_loader.subtree("pre_mixer_norm").is_ok() {
                Some(
                    Normalization::new(
                        ctx,
                        intermediate_data_type,
                        norm_config.clone(),
                        &layer_loader.subtree("pre_mixer_norm").unwrap(),
                    )
                    .expect("Failed to create pre-attention norm kernel"),
                )
            } else {
                None
            }
        } else {
            None
        };

        let qkv_projection = <dyn Linear<B>>::new(
            &attention_config.qkv_projection_config,
            model_dim,
            [num_heads * head_dim, num_groups * head_dim, num_groups * head_dim],
            ctx,
            &layer_loader.subtree("mixer.qkv_projection").unwrap(),
        )
        .expect("Failed to create qkv projection");

        let qk_norm = if attention_config.query_norm_config.is_some() || attention_config.key_norm_config.is_some() {
            match QKNorm::new(
                ctx,
                intermediate_data_type,
                attention_config.query_norm_config.clone(),
                attention_config.key_norm_config.clone(),
                &layer_loader.subtree("mixer").unwrap(),
                num_heads,
                num_groups,
                head_dim,
            ) {
                Ok(norm) => Some(norm),
                Err(e) => panic!("Failed to create QK norm kernel for layer {}: {:?}", layer_index, e),
            }
        } else {
            None
        };

        let out_projection = <dyn Linear<B>>::new(
            &attention_config.out_projection_config,
            num_heads * head_dim,
            [model_dim],
            ctx,
            &layer_loader.subtree("mixer.out_projection").unwrap(),
        )
        .expect("Failed to create out projection");

        let post_attention_norm = if let Some(norm_config) = &layer_config.post_attention_norm_config {
            Some(
                Normalization::new(
                    ctx,
                    intermediate_data_type,
                    norm_config.clone(),
                    &layer_loader.subtree("post_mixer_norm").unwrap(),
                )
                .expect("Failed to create post-attention norm kernel"),
            )
        } else {
            None
        };

        let mixer_residual_add = TensorAddSwap::<B>::new(ctx, intermediate_data_type).unwrap();

        let pre_mlp_norm = Normalization::new(
            ctx,
            intermediate_data_type,
            layer_config.pre_mlp_norm_config.clone(),
            &layer_loader.subtree("pre_mlp_norm").unwrap(),
        )
        .expect("Failed to create pre-MLP norm kernel");

        let (mlp, mlp_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            model_dim,
            hidden_dim,
            context.as_ref(),
            &layer_loader.subtree("mlp").unwrap(),
        )
        .expect("Failed to create mlp block");

        assert!(mlp_hadamard_factors.is_none(), "classifier doesn't support hadamard");

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(
                Normalization::new(
                    ctx,
                    intermediate_data_type,
                    norm_config.clone(),
                    &layer_loader.subtree("post_mlp_norm").unwrap(),
                )
                .expect("Failed to create post-MLP norm kernel"),
            )
        } else {
            None
        };

        let attention = Attention::new(
            ctx,
            intermediate_data_type,
            attention_scale,
            attention_config.has_sinks,
            false,
            attention_config.sliding_window_size,
            false,
        )
        .expect("Failed to create attention kernel");

        let mlp_residual_add = TensorAddSwap::<B>::new(ctx, intermediate_data_type).unwrap();

        Self {
            pre_attention_norm,
            qkv_projection,
            qk_norm,
            rope,
            attention,
            out_projection,
            post_attention_norm,
            mixer_residual_add,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
            mlp_residual_add,
            model_dim,
            num_heads,
            num_groups,
            head_dim,
        }
    }

    pub fn encode(
        &self,
        args: LayerArguments<'_, B>,
        parameters: &EncodingParameters,
        main: Allocation<B>,
        shortcut: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let LayerArguments {
            context,
            batch_dim,
            token_positions,
            token_subtrie_ranges,
            attention_sinks,
            rope_cosines,
            rope_sines,
            rope_max_sequence_length,
            rope_dim,
            #[cfg(feature = "tracing")]
            trace,
            ..
        } = args;
        #[cfg(feature = "tracing")]
        let layer_traces = trace;
        let layer_len = batch_dim * self.model_dim;

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.inputs);
        }

        debug_assert_eq!(main.as_buffer_range().1.len(), shortcut.as_buffer_range().1.len());
        encoder.encode_copy_allocation(&main, shortcut);

        let mut main = if let Some(ref pre_attn_norm) = self.pre_attention_norm {
            pre_attn_norm.encode(&main, 0, batch_dim, encoder)?
        } else {
            main
        };
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.pre_attention_norm);
        }

        let mut qkv = self.qkv_projection.encode(context, &mut main, batch_dim, encoder)?;
        if let Some(ref qk_norm) = self.qk_norm {
            qk_norm.encode(&mut qkv, batch_dim, encoder)?;
        }
        let cosines = rope_cosines.expect("Classifier attention layer requires RoPE cosine allocation");
        let sines = rope_sines.expect("Classifier attention layer requires RoPE sine allocation");
        let (queries, rotated_keys) = self.rope.encode(
            &qkv,
            token_positions,
            cosines,
            sines,
            batch_dim,
            self.num_heads,
            self.num_groups,
            self.head_dim,
            rope_max_sequence_length,
            rope_dim,
            encoder,
        )?;
        let mut attention_output = self.attention.encode(
            AttentionArguments {
                context,
                projection_step: parameters.projection_step.unwrap_or(0),
                token_subtrie_ranges,
                attention_sinks,
                kv_cache_layer: None,
            },
            &qkv,
            &queries,
            rotated_keys,
            None,
            batch_dim,
            self.num_heads,
            self.num_groups,
            self.head_dim,
            encoder,
        )?;
        main = self.out_projection.encode(context, &mut attention_output, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.attention);
        }

        if let Some(ref post_attn_norm) = self.post_attention_norm {
            main = post_attn_norm.encode(&main, 0, batch_dim, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                encoder.encode_copy_allocation(&main, &layer_traces.post_attention_norm);
            }
        }

        self.mixer_residual_add.encode(shortcut, &mut main, layer_len, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.mlp_inputs);
        }

        debug_assert_eq!(main.as_buffer_range().1.len(), shortcut.as_buffer_range().1.len());
        encoder.encode_copy_allocation(&main, shortcut);

        main = self.pre_mlp_norm.encode(&main, 0, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.pre_mlp_norm);
        }

        main = self.mlp.encode(context, &mut main, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.mlp);
        }

        if let Some(ref post_mlp_norm) = self.post_mlp_norm {
            main = post_mlp_norm.encode(&main, 0, batch_dim, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                encoder.encode_copy_allocation(&main, &layer_traces.post_mlp_norm);
            }
        }

        self.mlp_residual_add.encode(shortcut, &mut main, layer_len, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            encoder.encode_copy_allocation(&main, &layer_traces.outputs);
        }

        Ok(main)
    }

    pub fn rope_type(&self) -> RopeType {
        self.rope.rope_type()
    }
}
