use half::bf16;
use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{
            CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu, Kernels,
            kernel::TensorAddScaleKernel,
        },
    },
    config::{
        token_mixer::attention::AttentionConfig,
        vision::{VisionLayerConfig, VisionRoPEConfig},
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{LinearMatmul, LinearMatmulError},
        mixer::attention::{Attention, AttentionEncodeError, AttentionNewError, rope::RoPE},
        mlp::{MlpEncodeError, UngatedMlp, UngatedMlpNewError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum VisionLayerNewError<B: Backend> {
    #[error("Attention error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    LinearMatmul(#[from] LinearMatmulError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("MLP error: {0}")]
    UngatedMlp(#[from] UngatedMlpNewError<B>),
}

#[derive(Debug, Error)]
pub enum VisionLayerEncodeError<B: Backend> {
    #[error("Attention error: {0}")]
    Attention(#[from] AttentionEncodeError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
}

pub struct VisionLayer<B: Backend> {
    pre_attention_norm: Normalization<B>,
    attention: Attention<B>,
    pre_mlp_norm: Normalization<B>,
    mlp: UngatedMlp<B>,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    model_dim: u32,
}

impl<B: Backend> VisionLayer<B> {
    pub async fn new(
        data_type: DataType,
        config: &VisionLayerConfig,
        rope_config: &VisionRoPEConfig,
        parameter_tree: &ParameterTree<'_, B>,
        model_dim: u32,
        hidden_dim: u32,
        context: &B::Context,
    ) -> Result<Self, VisionLayerNewError<B>> {
        let padded_head_dim = config.attention_config.head_dim.next_power_of_two().max(64);
        let attention = Self::load_padded_attention::<bf16>(
            data_type,
            &config.attention_config,
            rope_config,
            &parameter_tree.subtree("attention"),
            model_dim,
            padded_head_dim,
            context,
        )
        .await?;

        let pre_attention_norm = Normalization::new(
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.pre_attention_norm_config,
            &parameter_tree.subtree("pre_attention_norm"),
            context,
        )
        .await?;

        let pre_mlp_norm = Normalization::new(
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.pre_mlp_norm_config,
            &parameter_tree.subtree("pre_mlp_norm"),
            context,
        )
        .await?;

        let mlp = UngatedMlp::new(
            data_type,
            &config.mlp_config,
            &parameter_tree.subtree("mlp"),
            model_dim,
            hidden_dim,
            model_dim,
            context,
        )
        .await?;

        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, true)
            .map_err(VisionLayerNewError::Backend)?;

        Ok(Self {
            pre_attention_norm,
            attention,
            pre_mlp_norm,
            mlp,
            residual_add,
            model_dim,
        })
    }

    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionLayerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("vision layer");
        let elements = row_count * self.model_dim;

        let hidden = self
            .pre_attention_norm
            .encode(&stream, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(VisionLayerEncodeError::Backend)?;

        let attention = {
            let command_buffer = &mut *command_buffer.debug_group("attention");
            self.attention.attend(hidden, &BatchTopology::flat(row_count), None, command_buffer)?
        };

        self.residual_add.encode(
            None::<&B::ScratchBuffer>,
            Some(&attention),
            &mut stream,
            elements,
            elements,
            1.0,
            command_buffer,
        );

        let hidden = self
            .pre_mlp_norm
            .encode(&stream, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(VisionLayerEncodeError::Backend)?;

        let mlp = self.mlp.encode(hidden, row_count, command_buffer)?;

        self.residual_add.encode(
            None::<&B::ScratchBuffer>,
            Some(&mlp),
            &mut stream,
            elements,
            elements,
            1.0,
            command_buffer,
        );

        Ok(stream)
    }

    async fn load_padded_attention<T: DataTypeRepr>(
        data_type: DataType,
        config: &AttentionConfig,
        rope_config: &VisionRoPEConfig,
        tree: &ParameterTree<'_, B>,
        model_dim: u32,
        padded_head_dim: u32,
        context: &B::Context,
    ) -> Result<Attention<B>, VisionLayerNewError<B>> {
        let qkv_heads = config.num_heads * (1 + u32::from(config.has_gate))
            + if config.is_kv_sharing {
                0
            } else {
                2 * config.num_groups
            };

        let qkv_proj_tree = tree.subtree("qkvg_projection");

        let qkv_weights_slice = qkv_proj_tree
            .subtree("weights")
            .read_slice::<T>("weights", &[qkv_heads * config.head_dim, model_dim])
            .await?;
        let qkv_weights_buf = get_padded_buffer::<B, T>(
            context,
            &qkv_weights_slice,
            config.head_dim as usize,
            padded_head_dim as usize,
            model_dim as usize,
        )
        .map_err(VisionLayerNewError::Backend)?;

        let qkv_biases_buf = if config.has_qkvg_biases {
            let qkv_biases_slice = qkv_proj_tree.read_slice::<T>("biases", &[qkv_heads * config.head_dim]).await?;
            let qkv_biases_buf = get_padded_buffer::<B, T>(
                context,
                &qkv_biases_slice,
                config.head_dim as usize,
                padded_head_dim as usize,
                1usize,
            )
            .map_err(VisionLayerNewError::Backend)?;
            Some(qkv_biases_buf)
        } else {
            None
        };

        let projection = LinearMatmul::from_dense(
            model_dim,
            qkv_heads * padded_head_dim,
            data_type,
            qkv_weights_buf,
            qkv_biases_buf,
            context,
        )?;

        let out_proj_tree = tree.subtree("out_projection");

        let out_weights_slice = out_proj_tree
            .subtree("weights")
            .read_slice::<T>("weights", &[model_dim, config.num_heads * config.head_dim])
            .await?;
        let out_weights_buf = get_padded_buffer::<B, T>(
            context,
            &out_weights_slice,
            config.head_dim as usize,
            padded_head_dim as usize,
            1usize,
        )
        .map_err(VisionLayerNewError::Backend)?;

        let out_biases_buf: Option<B::GlobalGpuBuffer> = if config.has_out_biases {
            let buffer = out_proj_tree.read_buffer("biases", &[model_dim], data_type).await?;
            Some(buffer)
        } else {
            None
        };

        let out_proj = LinearMatmul::from_dense(
            config.num_heads * padded_head_dim,
            model_dim,
            data_type,
            out_weights_buf,
            out_biases_buf,
            context,
        )?;

        let mut padded_config = config.clone();
        padded_config.head_dim = padded_head_dim;
        padded_config.scale = Some(config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()));

        let rope = vision_rope(rope_config, context).map_err(VisionLayerNewError::Backend)?;
        let attention = Attention::with_projections(
            data_type,
            Some(rope),
            &padded_config,
            Box::new(projection),
            Box::new(out_proj),
            tree,
            context,
        )
        .await?;

        Ok(attention)
    }
}

fn vision_rope<B: Backend>(
    rope_config: &VisionRoPEConfig,
    context: &B::Context,
) -> Result<RoPE<B>, B::Error> {
    let half_dim = rope_config.head_dim / 2;
    let frequency_count = rope_config.head_dim / 4;
    // inv_freq = 1 / base ^ (arange(0, half_dim, 2) / half_dim)
    let inverse_frequencies: Vec<f32> = (0..half_dim)
        .map(|pair| 1.0 / rope_config.base.powf((2 * (pair % frequency_count)) as f32 / half_dim as f32))
        .collect();
    RoPE::from_inverse_frequencies(&inverse_frequencies, 1.0, rope_config.head_dim, u32::MAX, context)
}

fn get_padded_buffer<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    data: &[T],
    head_dim: usize,
    padded_head_dim: usize,
    unit: usize,
) -> Result<B::GlobalGpuBuffer, B::Error> {
    let head_len = head_dim * unit;
    let padded_head_len = padded_head_dim * unit;
    let head_count = data.len() / head_len;

    let mut buffer = context.create_buffer_cpu_gpu(head_count * padded_head_len * size_of::<T>())?;
    let padded_heads = buffer.as_slice_mut::<T>().chunks_exact_mut(padded_head_len);
    for (padded_head, head) in padded_heads.zip(data.chunks_exact(head_len)) {
        let (values, padding) = padded_head.split_at_mut(head_len);
        values.copy_from_slice(head);
        padding.fill(T::zeroed());
    }
    buffer.handoff()
}
