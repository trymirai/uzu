use std::iter;

use derive_more::Debug;
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
        vision::VisionLayerConfig,
        weight_matrix::{AnyWeightMatrixSpec, Layout},
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{LinearMatmul, LinearMatmulError},
        mixer::{
            Mixer, MixerEncodeError,
            attention::{Attention, AttentionNewError, rope::PrecalculatedRoPE},
        },
        mlp::{Mlp, MlpEncodeError, UngatedMlp, UngatedMlpNewError},
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
    Attention(#[from] MixerEncodeError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
}

pub struct VisionLayer<B: Backend> {
    model_dim: u32,
    pre_attention_norm: Normalization<B>,
    attention: Attention<B>,
    pre_mlp_norm: Normalization<B>,
    mlp: UngatedMlp<B>,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> VisionLayer<B> {
    pub async fn new(
        model_dim: u32,
        hidden_dim: u32,
        data_type: DataType,
        config: &VisionLayerConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, VisionLayerNewError<B>> {
        let attention_tree = parameter_tree.subtree("attention");
        let attention = match data_type {
            DataType::BF16 => {
                padded_attention::<B, bf16>(model_dim, data_type, &config.attention_config, &attention_tree, context)
                    .await?
            },
            DataType::F32 => {
                padded_attention::<B, f32>(model_dim, data_type, &config.attention_config, &attention_tree, context)
                    .await?
            },
            data_type => return Err(LinearMatmulError::UnsupportedDataType(data_type).into()),
        };

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
            model_dim,
            hidden_dim,
            model_dim,
            data_type,
            &config.mlp_config,
            &parameter_tree.subtree("mlp"),
            context,
        )
        .await?;

        // In place: stream += update.
        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, true)
            .map_err(VisionLayerNewError::Backend)?;

        Ok(Self {
            model_dim,
            pre_attention_norm,
            attention,
            pre_mlp_norm,
            mlp,
            residual_add,
        })
    }

    /// `x + attention(norm(x))`, then `x + mlp(norm(x))`, over the `row_count` patches of one image.
    /// The attention is non-causal over all rows, so each call must hold a single image (frame).
    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        rope: &PrecalculatedRoPE<B>,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionLayerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("vision layer");
        let elements = row_count * self.model_dim;

        let hidden = self
            .pre_attention_norm
            .encode(&stream, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(VisionLayerEncodeError::Backend)?;
        let attention =
            self.attention.encode(hidden, Some(rope), &BatchTopology::flat(row_count, true), None, command_buffer)?;
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
}

/// Loads the attention with every head zero-padded to `padded_head_dim` in the projection weights.
/// It applies RoPE from tables passed to `encode` (see `rope::vision_rope`), with no context length limit.
async fn padded_attention<B: Backend, T: DataTypeRepr>(
    model_dim: u32,
    data_type: DataType,
    config: &AttentionConfig,
    tree: &ParameterTree<'_, B>,
    context: &B::Context,
) -> Result<Attention<B>, VisionLayerNewError<B>> {
    assert!(
        config.query_norm_config.is_none() && config.key_norm_config.is_none() && !config.normalize_values,
        "per-head norms can't run on padded heads"
    );

    let qkv = tree.subtree("qkvg_projection");
    let out = tree.subtree("out_projection");
    for projection in [&qkv, &out] {
        let spec = projection.subtree("weights").metadata::<AnyWeightMatrixSpec>("spec")?;
        assert!(
            matches!(&spec, AnyWeightMatrixSpec::FullPrecisionSpec(spec) if spec.layout == Layout::OutputInput),
            "padded heads need full-precision output-input weights, got {spec:?}"
        );
    }

    // The attention kernels run on power-of-two head dims (Hadamard-rotated int8 Q/K/V), so other head dims are zero-padded at load.
    // Zero channels don't change the scores, and the padded output channels meet zero columns.
    let padded_head_dim = config.head_dim.next_power_of_two().max(64);

    // Each head is a block of `head_dim * unit` values: qkv rows of `model_dim`, single biases, or out columns.
    let pad = |data: Box<[T]>, unit: u32| -> Vec<T> {
        let padded_len = (padded_head_dim * unit) as usize;
        data.chunks_exact((config.head_dim * unit) as usize)
            .flat_map(|head| head.iter().copied().chain(iter::repeat_n(T::zeroed(), padded_len - head.len())))
            .collect()
    };
    let upload = |data: Vec<T>| -> Result<B::GlobalGpuBuffer, VisionLayerNewError<B>> {
        let mut buffer =
            context.create_buffer_cpu_gpu(size_of_val(data.as_slice())).map_err(VisionLayerNewError::Backend)?;
        buffer.as_slice_mut::<T>().copy_from_slice(&data);
        buffer.handoff().map_err(VisionLayerNewError::Backend)
    };

    let qkv_heads = config.num_heads * (1 + u32::from(config.has_gate))
        + if config.is_kv_sharing {
            0
        } else {
            2 * config.num_groups
        };
    let qkv_weights =
        qkv.subtree("weights").read_slice::<T>("weights", &[qkv_heads * config.head_dim, model_dim]).await?;
    let qkv_biases = if config.has_qkvg_biases {
        Some(upload(pad(qkv.read_slice::<T>("biases", &[qkv_heads * config.head_dim]).await?, 1))?)
    } else {
        None
    };
    let projection = LinearMatmul::from_dense(
        model_dim,
        qkv_heads * padded_head_dim,
        data_type,
        upload(pad(qkv_weights, model_dim))?,
        qkv_biases,
        context,
    )?;

    let out_weights =
        out.subtree("weights").read_slice::<T>("weights", &[model_dim, config.num_heads * config.head_dim]).await?;
    let out_biases = if config.has_out_biases {
        Some(out.read_buffer("biases", &[model_dim], data_type).await?)
    } else {
        None
    };
    let out_projection = LinearMatmul::from_dense(
        config.num_heads * padded_head_dim,
        model_dim,
        data_type,
        upload(pad(out_weights, 1))?,
        out_biases,
        context,
    )?;

    // Keep the scale of the unpadded head dim.
    let mut padded_config = config.clone();
    padded_config.head_dim = padded_head_dim;
    padded_config.scale = Some(config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()));
    let attention = Attention::with_projections(
        data_type,
        true,
        None,
        &padded_config,
        Box::new(projection),
        Box::new(out_projection),
        tree,
        context,
    )
    .await?;
    Ok(attention)
}
