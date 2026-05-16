use std::rc::Rc;

use crate::{
    DataType,
    backends::common::{Allocation, AsBufferRangeRef, Backend, Encoder, Kernels, kernel::TensorAddSwapKernel},
    config::{TransformerConfig, TransformerLayerConfig},
    encodable_block::{
        Attention, AttentionArguments, LayerArguments, Linear, Mlp, Normalization, QKVNorm, QkUnpack, Rope,
    },
    parameters::ParameterTree,
};

pub struct ClassifierLayer<B: Backend> {
    pre_attention_norm: Option<Normalization<B>>,
    qkv_projection: Box<dyn Linear<B>>,
    qkv_norm: Option<QKVNorm<B>>,
    rope: Rc<Rope<B>>,
    qk_unpack: Rc<QkUnpack<B>>,
    attention: Attention<B>,
    out_projection: Box<dyn Linear<B>>,
    post_attention_norm: Option<Normalization<B>>,
    mixer_residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    pre_mlp_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
    post_mlp_norm: Option<Normalization<B>>,
    mlp_residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    model_dim: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
}

impl<B: Backend> ClassifierLayer<B> {
    pub fn new(
        context: &B::Context,
        transformer_config: &TransformerConfig,
        layer_config: &TransformerLayerConfig,
        layer_index: usize,
        layer_loader: &ParameterTree<B::Context>,
        rope: Rc<Rope<B>>,
        qk_unpack: Rc<QkUnpack<B>>,
    ) -> Self {
        let attention_config = layer_config.mixer_config.as_attention().expect("Classifier layers must use attention");
        let intermediate_data_type: DataType = attention_config.qkv_projection_config.activation_precision().into();

        let pre_attention_norm = layer_config.pre_mixer_norm_config.as_ref().map(|norm_config| {
            Normalization::new(
                context,
                intermediate_data_type,
                norm_config.clone(),
                &layer_loader.subtree("pre_mixer_norm").unwrap(),
            )
            .expect("Failed to create pre-attention norm kernel")
        });

        let qkv_projection = <dyn Linear<B>>::new(
            &attention_config.qkv_projection_config,
            transformer_config.model_dim,
            [
                attention_config.num_heads * attention_config.head_dim,
                attention_config.num_groups * attention_config.head_dim,
                attention_config.num_groups * attention_config.head_dim,
            ],
            context,
            &layer_loader.subtree("mixer.qkv_projection").unwrap(),
        )
        .expect("Failed to create qkv projection");

        let value_norm_config = attention_config.value_norm_config();
        let qkv_norm = if attention_config.query_norm_config.is_some()
            || attention_config.key_norm_config.is_some()
            || value_norm_config.is_some()
        {
            match QKVNorm::new(
                context,
                intermediate_data_type,
                attention_config.query_norm_config.clone(),
                attention_config.key_norm_config.clone(),
                value_norm_config,
                &layer_loader.subtree("mixer").unwrap(),
                attention_config.num_heads,
                attention_config.num_groups,
                attention_config.head_dim,
            ) {
                Ok(qkv_norm) => Some(qkv_norm),
                Err(error) => panic!("Failed to create QKV norm kernel for layer {}: {:?}", layer_index, error),
            }
        } else {
            None
        };

        let out_projection = <dyn Linear<B>>::new(
            &attention_config.out_projection_config,
            attention_config.num_heads * attention_config.head_dim,
            [transformer_config.model_dim],
            context,
            &layer_loader.subtree("mixer.out_projection").unwrap(),
        )
        .expect("Failed to create out projection");

        let post_attention_norm = layer_config.post_mixer_norm_config.as_ref().map(|norm_config| {
            Normalization::new(
                context,
                intermediate_data_type,
                norm_config.clone(),
                &layer_loader.subtree("post_mixer_norm").unwrap(),
            )
            .expect("Failed to create post-attention norm kernel")
        });

        let mixer_residual_add = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, intermediate_data_type)
            .expect("Failed to create mixer residual add kernel");

        let pre_mlp_norm = Normalization::new(
            context,
            intermediate_data_type,
            layer_config.pre_mlp_norm_config.clone(),
            &layer_loader.subtree("pre_mlp_norm").unwrap(),
        )
        .expect("Failed to create pre-MLP norm kernel");

        let (mlp, mlp_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            transformer_config.model_dim,
            transformer_config.hidden_dim,
            context,
            &layer_loader.subtree("mlp").unwrap(),
        )
        .expect("Failed to create mlp block");

        assert!(mlp_hadamard_factors.is_none(), "classifier doesn't support hadamard");

        let post_mlp_norm = layer_config.post_mlp_norm_config.as_ref().map(|norm_config| {
            Normalization::new(
                context,
                intermediate_data_type,
                norm_config.clone(),
                &layer_loader.subtree("post_mlp_norm").unwrap(),
            )
            .expect("Failed to create post-MLP norm kernel")
        });

        let attention = Attention::new(context, intermediate_data_type, attention_config, false)
            .expect("Failed to create attention kernel");

        let mlp_residual_add = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, intermediate_data_type)
            .expect("Failed to create mlp residual add kernel");

        Self {
            pre_attention_norm,
            qkv_projection,
            qkv_norm,
            rope,
            qk_unpack,
            attention,
            out_projection,
            post_attention_norm,
            mixer_residual_add,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
            mlp_residual_add,
            model_dim: transformer_config.model_dim,
            num_heads: attention_config.num_heads,
            num_groups: attention_config.num_groups,
            head_dim: attention_config.head_dim,
        }
    }

    pub fn encode(
        &self,
        args: LayerArguments<B>,
        main: Allocation<B>,
        shortcut: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let LayerArguments {
            batch_dim,
            token_positions,
            token_subtrie_ranges,
            attention_sinks,
            rope_buffers,
            #[cfg(feature = "tracing")]
            trace,
            ..
        } = args;
        #[cfg(feature = "tracing")]
        let mut layer_traces = trace;
        let layer_len = batch_dim * self.model_dim;

        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.inputs.allocation_mut(), ..);
        }

        debug_assert_eq!(main.as_buffer_range_ref().range().len(), shortcut.as_buffer_range_ref().range().len());
        encoder.encode_copy(&main, .., shortcut, ..);

        let mut main = if let Some(ref pre_attn_norm) = self.pre_attention_norm {
            pre_attn_norm.encode(&main, 0, batch_dim, encoder)?
        } else {
            main
        };
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.pre_attention_norm.allocation_mut(), ..);
        }

        let mut qkv = self.qkv_projection.encode(main, batch_dim, encoder)?;
        if let Some(ref qkv_norm) = self.qkv_norm {
            qkv_norm.encode(&mut qkv, batch_dim, encoder)?;
        }
        let (queries, rotated_keys) = match rope_buffers {
            Some(rope_buffers) => self.rope.encode(
                &qkv,
                token_positions,
                &rope_buffers.cosines,
                &rope_buffers.sines,
                batch_dim,
                self.num_heads,
                self.num_groups,
                self.head_dim,
                rope_buffers.max_sequence_length(),
                rope_buffers.dim(),
                encoder,
            )?,
            None => self.qk_unpack.encode(&qkv, batch_dim, self.num_heads, self.num_groups, self.head_dim, encoder)?,
        };
        let attention_output = self.attention.encode(
            AttentionArguments {
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
        main = self.out_projection.encode(attention_output, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.attention.allocation_mut(), ..);
        }

        if let Some(ref post_attn_norm) = self.post_attention_norm {
            main = post_attn_norm.encode(&main, 0, batch_dim, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&main, .., layer_traces.post_attention_norm.allocation_mut(), ..);
            }
        }

        self.mixer_residual_add.encode(&mut *shortcut, &mut main, layer_len as u32, encoder);
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.mlp_inputs.allocation_mut(), ..);
        }

        debug_assert_eq!(main.as_buffer_range_ref().range().len(), shortcut.as_buffer_range_ref().range().len());
        encoder.encode_copy(&main, .., shortcut, ..);

        main = self.pre_mlp_norm.encode(&main, 0, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.pre_mlp_norm.allocation_mut(), ..);
        }

        main = self.mlp.encode(main, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.mlp.allocation_mut(), ..);
        }

        if let Some(ref post_mlp_norm) = self.post_mlp_norm {
            main = post_mlp_norm.encode(&main, 0, batch_dim, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&main, .., layer_traces.post_mlp_norm.allocation_mut(), ..);
            }
        }

        self.mlp_residual_add.encode(shortcut, &mut main, layer_len as u32, encoder);
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.outputs.allocation_mut(), ..);
        }

        Ok(main)
    }
}
