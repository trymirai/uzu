use std::sync::Arc;

use derive_more::Debug;
use futures::future::try_join_all;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, kernel::ActivationTransform},
    config::{
        residual::AnyResidualConfig,
        transformer_layer::{SublayerConfigCommon, TransformerLayerConfig, TransformerLayerConvConfig},
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        convolution::{ConvolutionNewError, SeparableCausalConv},
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mixer::{Mixer, MixerEncodeError, MixerNewError, MixerState, attention::rope::PrecalculatedRoPE},
        mlp::{Mlp, MlpBlockNewError, MlpEncodeError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::PerLayerEmbeddingError,
        residual::{ResidualConnection, ResidualNewError},
        stream_op::{StreamOp, StreamOpNewError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

#[derive(Debug, Error)]
pub enum TransformerLayerError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Mixer error: {0}")]
    Mixer(#[from] MixerNewError<B>),
    #[error("Mixer encoding error: {0}")]
    MixerEncode(#[from] MixerEncodeError<B>),
    #[error("MLP error: {0}")]
    MlpBlock(#[from] MlpBlockNewError<B>),
    #[error("MLP encoding error: {0}")]
    MlpEncode(#[from] MlpEncodeError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Convolution error: {0}")]
    Convolution(#[from] ConvolutionNewError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Residual error: {0}")]
    Residual(#[from] ResidualNewError<B>),
    #[error("Stream op error: {0}")]
    StreamOp(#[from] StreamOpNewError<B>),
    #[error("Layer {layer_index} sets a residual scale op but has no mlp post_norm")]
    ResidualScaleWithoutPostMlpNorm {
        layer_index: u32,
    },
    #[error("Layer {layer_index} uses prologue stream ops, which are not supported")]
    UnsupportedPrologueOps {
        layer_index: u32,
    },
    #[error("Layer {layer_index} uses an n-gram injection op, which is not supported")]
    UnsupportedNGramInjection {
        layer_index: u32,
    },
    #[error("Layer {layer_index} has multiple per-layer embedding injection ops, but only one is applied")]
    MultiplePerLayerEmbeddingInjectionOps {
        layer_index: u32,
    },
    #[error("Layer {layer_index} has multiple residual scale ops, but only one is applied")]
    MultipleResidualScaleOps {
        layer_index: u32,
    },
    #[error("Layer {layer_index} {sublayer} body expects RHT-rotated input but has no pre_norm or conv to apply it")]
    UnappliedInputRht {
        layer_index: u32,
        sublayer: &'static str,
    },
}

struct TransformerLayerConv<B: Backend> {
    model_dim: u32,
    pre_conv: SeparableCausalConv<B>,
    kernel_projection: Box<dyn Linear<B>>,
    post_conv: SeparableCausalConv<B>,
    input_rht: Option<(ActivationTransform<B>, B::GlobalGpuBuffer)>,
    coefficient_count: u32,
}

struct Sublayer<B: Backend> {
    model_dim: u32,
    pre_norm: Option<Normalization<B>>,
    conv: Option<TransformerLayerConv<B>>,
    post_norm: Option<Normalization<B>>,
    connection: ResidualConnection<B>,
}

pub struct TransformerLayer<B: Backend> {
    pub layer_index: u32,
    pub kv_source_layer_index: Option<u32>,
    prologue: Box<[StreamOp<B>]>,
    pub mixer: Box<dyn Mixer<B>>,
    mixer_sublayer: Sublayer<B>,
    pub mlp: Box<dyn Mlp<B>>,
    mlp_sublayer: Sublayer<B>,
    epilogue: Box<[StreamOp<B>]>,
}

impl<B: Backend> TransformerLayerConv<B> {
    async fn new(
        model_dim: u32,
        data_type: DataType,
        input_hadamard_factors: Option<B::GlobalGpuBuffer>,
        config: &TransformerLayerConvConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let pre_conv = SeparableCausalConv::new(
            model_dim,
            config.conv_kernel_size,
            config.conv_group_size,
            data_type,
            &config.conv_config,
            &parameter_tree.subtree("pre_conv"),
            context,
        )
        .await?;
        let post_conv = SeparableCausalConv::new(
            model_dim,
            config.conv_kernel_size,
            config.conv_group_size,
            data_type,
            &config.conv_config,
            &parameter_tree.subtree("post_conv"),
            context,
        )
        .await?;

        let coefficient_count = config.conv_kernel_size * (model_dim / config.conv_group_size);
        let projection_dim = coefficient_count * 2;
        let kernel_projection = <dyn Linear<B>>::new(
            model_dim,
            projection_dim,
            data_type,
            false,
            &parameter_tree.subtree("kernel_projection"),
            context,
        )
        .await?;

        let input_rht = input_hadamard_factors
            .map(|factors| ActivationTransform::input_rht(context, data_type, true).map(|kernel| (kernel, factors)))
            .transpose()
            .map_err(TransformerLayerError::Backend)?;

        Ok(Self {
            model_dim,
            pre_conv,
            kernel_projection,
            post_conv,
            input_rht,
            coefficient_count,
        })
    }

    fn encode_pre_convolution(
        &self,
        input: impl BufferRef<Backend = B>,
        sequence_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), TransformerLayerError<B>> {
        let mut projection_input =
            command_buffer.allocate_scratch(input.size()).map_err(TransformerLayerError::Backend)?;
        command_buffer.encode_copy(input, &mut projection_input);
        let coefficients = self.kernel_projection.encode(projection_input, sequence_length, command_buffer)?;
        let mut output = self
            .pre_conv
            .encode(input, &coefficients, 2 * self.coefficient_count, 0, sequence_length, command_buffer)
            .map_err(TransformerLayerError::Backend)?;
        if let Some((transform, factors)) = &self.input_rht {
            transform.encode_fp_in_place(&mut output, factors, None, sequence_length, self.model_dim, command_buffer);
        }
        Ok((output, coefficients))
    }

    fn encode_post_convolution(
        &self,
        input: impl BufferRef<Backend = B>,
        coefficients: impl BufferRef<Backend = B>,
        sequence_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.post_conv.encode(
            input,
            coefficients,
            2 * self.coefficient_count,
            self.coefficient_count,
            sequence_length,
            command_buffer,
        )
    }
}

impl<B: Backend> Sublayer<B> {
    async fn new(
        model_dim: u32,
        data_type: DataType,
        layer_index: u32,
        name: &'static str,
        config: &SublayerConfigCommon,
        mut input_hadamard_factors: Option<B::GlobalGpuBuffer>,
        residual_config: &AnyResidualConfig,
        layer_parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let parameter_tree = layer_parameter_tree.subtree(name);

        let conv = if let Some(conv_config) = &config.conv_config {
            Some(
                TransformerLayerConv::new(
                    model_dim,
                    data_type,
                    input_hadamard_factors.take(),
                    conv_config,
                    &parameter_tree.subtree("conv"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let pre_norm = if let Some(norm_config) = &config.pre_norm_config {
            Some(
                Normalization::new(
                    model_dim,
                    data_type,
                    data_type,
                    input_hadamard_factors.take().map(Arc::new),
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    norm_config,
                    &parameter_tree.subtree("pre_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        if input_hadamard_factors.is_some() {
            return Err(TransformerLayerError::UnappliedInputRht {
                layer_index,
                sublayer: name,
            });
        }

        let post_norm = if let Some(norm_config) = &config.post_norm_config {
            Some(
                Normalization::new(
                    model_dim,
                    data_type,
                    data_type,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    norm_config,
                    &parameter_tree.subtree("post_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let connection =
            ResidualConnection::new(residual_config, data_type, &parameter_tree.subtree("connection"), context)?;

        Ok(Self {
            model_dim,
            pre_norm,
            conv,
            post_norm,
            connection,
        })
    }

    fn encode(
        &self,
        stream: B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        body: impl FnOnce(
            B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<B::ScratchBuffer, TransformerLayerError<B>>,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let mut hidden = if let Some(pre_norm) = &self.pre_norm {
            pre_norm
                .encode(&stream, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerLayerError::Backend)?
        } else {
            let mut hidden = command_buffer.allocate_scratch(stream.size()).map_err(TransformerLayerError::Backend)?;
            command_buffer.encode_copy(&stream, &mut hidden);
            hidden
        };

        if let Some(conv) = &self.conv {
            let (conv_output, coefficients) = conv.encode_pre_convolution(&hidden, row_count, command_buffer)?;
            let body_output = body(conv_output, command_buffer)?;
            hidden = conv
                .encode_post_convolution(&body_output, &coefficients, row_count, command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        } else {
            hidden = body(hidden, command_buffer)?;
        }

        if let Some(post_norm) = &self.post_norm {
            hidden = post_norm
                .encode(&hidden, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }

        let update = hidden;
        self.connection
            .encode_update(&stream, &update, row_count * self.model_dim, command_buffer)
            .map_err(TransformerLayerError::Backend)
    }
}

impl<B: Backend> TransformerLayer<B> {
    pub async fn new(
        model_dim: u32,
        hidden_dim: u32,
        _num_layers: u32,
        layer_index: u32,
        data_type: DataType,
        layer_config: &TransformerLayerConfig,
        residual_config: &AnyResidualConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let prologue_params = parameter_tree.subtree("prologue");
        let prologue = try_join_all(layer_config.prologue_configs.iter().enumerate().map(|(i, cfg)| {
            StreamOp::new(model_dim, data_type, cfg, residual_config, prologue_params.subtree(&i.to_string()), context)
        }))
        .await?
        .into_boxed_slice();

        let epilogue_params = parameter_tree.subtree("epilogue");
        let epilogue = try_join_all(layer_config.epilogue_configs.iter().enumerate().map(|(i, cfg)| {
            StreamOp::new(model_dim, data_type, cfg, residual_config, epilogue_params.subtree(&i.to_string()), context)
        }))
        .await?
        .into_boxed_slice();

        let (mixer, mixer_hadamard_factors) = <dyn Mixer<B>>::new(
            model_dim,
            data_type,
            layer_config.rope_config.as_ref(),
            &layer_config.mixer.body_config,
            &parameter_tree.subtree("mixer").subtree("body"),
            context,
        )
        .await?;
        let mixer_sublayer = Sublayer::new(
            model_dim,
            data_type,
            layer_index,
            "mixer",
            &layer_config.mixer.common,
            mixer_hadamard_factors,
            residual_config,
            parameter_tree,
            context,
        )
        .await?;

        let (mlp, mlp_hadamard_factors) = <dyn Mlp<B>>::new(
            model_dim,
            layer_config.hidden_dim.unwrap_or(hidden_dim),
            data_type,
            &layer_config.mlp.body_config,
            &parameter_tree.subtree("mlp").subtree("body"),
            context,
        )
        .await?;
        let mlp_sublayer = Sublayer::new(
            model_dim,
            data_type,
            layer_index,
            "mlp",
            &layer_config.mlp.common,
            mlp_hadamard_factors,
            residual_config,
            parameter_tree,
            context,
        )
        .await?;

        Ok(Self {
            layer_index,
            kv_source_layer_index: layer_config.kv_source_layer_index,
            prologue,
            mixer,
            mixer_sublayer,
            mlp,
            mlp_sublayer,
            epilogue,
        })
    }

    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        _per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let command_buffer = &mut *command_buffer.debug_group(&format!("transformer layer {}", self.layer_index));
        let row_count = batch_dim.size();

        for op in &self.prologue {
            stream = op.encode(stream, batch_dim, command_buffer).map_err(TransformerLayerError::Backend)?;
        }

        stream = self.mixer_sublayer.encode(stream, row_count, command_buffer, |hidden, command_buffer| {
            Ok(self.mixer.encode(hidden, precalculated_rope, batch_dim, state, command_buffer)?)
        })?;

        stream = self.mlp_sublayer.encode(stream, row_count, command_buffer, |hidden, command_buffer| {
            Ok(self.mlp.encode(hidden, row_count, command_buffer)?)
        })?;

        for op in &self.epilogue {
            stream = op.encode(stream, batch_dim, command_buffer).map_err(TransformerLayerError::Backend)?;
        }

        Ok(stream)
    }
}
