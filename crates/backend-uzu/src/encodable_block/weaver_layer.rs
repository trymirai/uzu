use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AsBufferRangeMut, Backend, Encoder, Kernels,
        kernel::{AncestorAttentionKernel, AttentionPrepareKernel},
    },
    config::{
        activation::{AnyActivation, silu::SiLU},
        mlp::{AnyMLPConfig, dense_mlp::DenseMLPConfig},
        weaver::WeaverConfig,
    },
    encodable_block::{
        linear::Linear,
        mixer::attention::{
            core::{AttentionCoreNewArguments, AttentionCores},
            rope::PrecalculatedRoPE,
        },
        mlp::Mlp,
        normalization::{Normalization, PostLayerScalar, ShortcutMode},
        weaver::{DATA_TYPE, ROPE_DATA_TYPE, WeaverNewError},
    },
    parameters::ParameterTree,
};

pub struct PreparedPrefixAttention<B: Backend> {
    pub queries: Allocation<B>,
    pub kv_cache: Allocation<B>,
}

pub struct WeaverLayer<B: Backend> {
    pub pre_attention_norm: Normalization<B>,
    pub qkv_projection: Box<dyn Linear<B>>,
    attention_prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    pub prefix_attention: AttentionCores<B>,
    pub ancestor_attention: <B::Kernels as Kernels>::AncestorAttentionKernel,
    out_projection: Box<dyn Linear<B>>,
    pre_mlp_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
    pub attention_scale: f32,
    model_dim: u32,
    num_heads: u32,
    head_dim: u32,
    pub max_depth: u32,
}

impl<B: Backend> WeaverLayer<B> {
    pub fn new(
        context: &B::Context,
        config: &WeaverConfig,
        add_to_residual: bool,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, WeaverNewError<B>> {
        let WeaverConfig {
            model_dim,
            hidden_dim,
            num_heads,
            max_depth,
            ref norm_config,
            ref linear_config,
            ..
        } = *config;
        let head_dim = model_dim / num_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt();
        let qkv_projection = <dyn Linear<B>>::new(
            model_dim,
            [3 * model_dim],
            false,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("qkv_projection"),
        )?;
        let out_projection = <dyn Linear<B>>::new(
            model_dim,
            [model_dim],
            false,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("out_projection"),
        )?;
        let prefix_attention = AttentionCores::new(
            AttentionCoreNewArguments {
                head_dim,
                num_groups: num_heads,
                num_q_heads: num_heads,
                has_sinks: false,
                is_kv_cache_ring: false,
                is_causal: true,
                is_trie: false,
                sliding_window_size: None,
                scale: Some(attention_scale),
                data_type: DATA_TYPE,
            },
            context,
        )
        .map_err(WeaverNewError::Backend)?;
        let attention_prepare =
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(context, DATA_TYPE, ROPE_DATA_TYPE, true, true)
                .map_err(WeaverNewError::Backend)?;
        let pre_attention_norm = Normalization::new(
            model_dim,
            None,
            if add_to_residual {
                ShortcutMode::Add
            } else {
                ShortcutMode::Copy
            },
            PostLayerScalar::None,
            DATA_TYPE,
            norm_config,
            &parameter_tree.subtree("pre_attention_norm"),
            context,
        )?;
        let pre_mlp_norm = Normalization::new(
            model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            DATA_TYPE,
            norm_config,
            &parameter_tree.subtree("pre_mlp_norm"),
            context,
        )?;
        let mlp_config = AnyMLPConfig::DenseMLPConfig(DenseMLPConfig::unclipped(
            linear_config.clone(),
            AnyActivation::SiLU(SiLU::default()),
            true,
            true,
        ));
        let (mlp, up_input_hadamard_factors) =
            <dyn Mlp<B>>::new(&mlp_config, model_dim, hidden_dim, context, &parameter_tree.subtree("mlp"), DATA_TYPE)?;
        assert!(up_input_hadamard_factors.is_none(), "Weaver MLP does not support input Hadamard factors");
        let ancestor_attention = <B::Kernels as Kernels>::AncestorAttentionKernel::new(context, head_dim, num_heads)
            .map_err(WeaverNewError::Backend)?;
        Ok(Self {
            qkv_projection,
            out_projection,
            prefix_attention,
            attention_prepare,
            pre_attention_norm,
            pre_mlp_norm,
            mlp,
            ancestor_attention,
            attention_scale,
            model_dim,
            num_heads,
            head_dim,
            max_depth,
        })
    }

    pub fn encode_prefix_attention(
        &self,
        residual_input: &Allocation<B>,
        residual_state: &mut Allocation<B>,
        rope: &PrecalculatedRoPE<B>,
        token_count: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<PreparedPrefixAttention<B>, B::Error> {
        let attention_input =
            self.pre_attention_norm.encode(residual_input, 0, token_count, Some(residual_state), encoder)?;
        let qkv = self.qkv_projection.encode(attention_input, token_count, encoder)?;
        let mut queries =
            encoder.allocate_scratch_for_shape(&[self.num_heads, token_count, self.head_dim], DATA_TYPE)?;
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let mut kv_cache = encoder.allocate_scratch_for_shape(&[2, token_count, self.model_dim], DATA_TYPE)?;
        let (keys, values) = kv_cache.as_buffer_range_mut().split_at(kv_plane_bytes);
        self.attention_prepare.encode(
            &qkv,
            &mut queries,
            Some(keys),
            Some(values),
            Some(&rope.cosines),
            Some(&rope.sines),
            self.num_heads,
            Some(self.num_heads),
            self.head_dim,
            Some(rope.dim),
            Some(0),
            3 * self.model_dim,
            token_count,
            encoder,
        );
        Ok(PreparedPrefixAttention {
            queries,
            kv_cache,
        })
    }

    pub fn encode_post_attention(
        &self,
        attention_output: Allocation<B>,
        residual_state: &mut Allocation<B>,
        token_count: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let projected_attention = self.out_projection.encode(attention_output, token_count, encoder)?;
        let mlp_input =
            self.pre_mlp_norm.encode(&projected_attention, 0, token_count, Some(residual_state), encoder)?;
        self.mlp.encode(mlp_input, token_count, encoder)
    }
}
