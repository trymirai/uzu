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
        linear::{InputBasis, InputRhtParameters, Linear, LinearBlockError, LinearEncodeError},
        mixer::{Mixer, MixerEncodeError, MixerNewError, MixerState},
        mlp::{Mlp, MlpBlockNewError, MlpEncodeError, MlpStarted},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        residual::{ResidualConnection, ResidualEncodeError, ResidualNewError, ResidualStarted},
        stream_op::{StreamOp, StreamOpContext, StreamOpEncodeError, StreamOpNewError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{maybe_mut::MaybeMut, maybe_owned::MaybeOwned},
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
    #[error("Residual error: {0}")]
    Residual(#[from] ResidualNewError<B>),
    #[error("Residual encoding error: {0}")]
    ResidualEncode(#[from] ResidualEncodeError<B>),
    #[error("Stream op error: {0}")]
    StreamOp(#[from] StreamOpNewError<B>),
    #[error("Stream op encoding error: {0}")]
    StreamOpEncode(#[from] StreamOpEncodeError<B>),
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
    input_rht: Option<(ActivationTransform<B>, Arc<B::GlobalGpuBuffer>)>,
    coefficient_count: u32,
}

struct Sublayer<B: Backend> {
    pre_norm: Option<Normalization<B>>,
    conv: Option<TransformerLayerConv<B>>,
    post_norm: Option<Normalization<B>>,
    connection: ResidualConnection<B>,
}

struct SublayerStarted<B: Backend> {
    residual: ResidualStarted<B>,
    conv_coefficients: Option<B::ScratchBuffer>,
}

pub struct PendingMlp<'a, B: Backend> {
    mlp: MlpStarted<'a, B>,
    sublayer: SublayerStarted<B>,
}

impl<B: Backend> PendingMlp<'_, B> {
    pub fn abandon(
        self,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let _ = self.mlp.finish(command_buffer);
    }
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
        input_hadamard_factors: Option<InputRhtParameters<B>>,
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
            .map(|rht| {
                ActivationTransform::input_rht(context, data_type, rht.hadamard_size, true)
                    .map(|kernel| (kernel, rht.signs))
            })
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
        apply_input_rht: bool,
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
        if apply_input_rht && let Some((transform, signs)) = &self.input_rht {
            transform.encode_fp_in_place(
                &mut output,
                Some(&**signs),
                None,
                sequence_length,
                self.model_dim,
                command_buffer,
            );
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
        mut input_hadamard_factors: Option<InputRhtParameters<B>>,
        allow_unrotated_input: bool,
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
            let input_hadamard_factors = input_hadamard_factors.take();
            Some(if allow_unrotated_input {
                Normalization::new_with_unrotated_input(
                    model_dim,
                    data_type,
                    data_type,
                    input_hadamard_factors,
                    ShortcutMode::None,
                    norm_config,
                    &parameter_tree.subtree("pre_norm"),
                    context,
                )
                .await?
            } else {
                Normalization::new(
                    model_dim,
                    data_type,
                    data_type,
                    input_hadamard_factors,
                    ShortcutMode::None,
                    norm_config,
                    &parameter_tree.subtree("pre_norm"),
                    context,
                )
                .await?
            })
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
                    norm_config,
                    &parameter_tree.subtree("post_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let connection = ResidualConnection::new(
            residual_config,
            model_dim,
            data_type,
            &parameter_tree.subtree("connection"),
            context,
        )
        .await?;

        Ok(Self {
            pre_norm,
            conv,
            post_norm,
            connection,
        })
    }

    fn start(
        &self,
        stream: B::ScratchBuffer,
        row_count: u32,
        input_basis: InputBasis,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(SublayerStarted<B>, B::ScratchBuffer), TransformerLayerError<B>> {
        let mut conv_coefficients = None;
        let (residual, hidden) = self.connection.start_with_body_input::<TransformerLayerError<B>>(
            stream,
            row_count,
            command_buffer,
            |input, command_buffer| {
                let mut hidden = match (&self.pre_norm, input) {
                    (Some(pre_norm), input) => pre_norm
                        .encode_with_unrotated_input(
                            &*input,
                            0,
                            row_count,
                            None::<&mut B::ScratchBuffer>,
                            input_basis == InputBasis::AneUnrotated,
                            command_buffer,
                        )
                        .map_err(TransformerLayerError::Backend)?,
                    (None, MaybeOwned::Owned(input)) => input,
                    (None, MaybeOwned::Borrowed(input)) => {
                        let mut hidden =
                            command_buffer.allocate_scratch(input.size()).map_err(TransformerLayerError::Backend)?;
                        command_buffer.encode_copy(input, &mut hidden);
                        hidden
                    },
                };

                if let Some(conv) = &self.conv {
                    let (conv_output, coefficients) = conv.encode_pre_convolution(
                        &hidden,
                        row_count,
                        input_basis != InputBasis::AneUnrotated,
                        command_buffer,
                    )?;
                    hidden = conv_output;
                    conv_coefficients = Some(coefficients);
                }
                Ok(hidden)
            },
        )?;
        Ok((
            SublayerStarted {
                residual,
                conv_coefficients,
            },
            hidden,
        ))
    }

    fn finish(
        &self,
        started: SublayerStarted<B>,
        mut hidden: B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        if let Some(coefficients) = started.conv_coefficients {
            hidden = self
                .conv
                .as_ref()
                .expect("convolution coefficients require convolution")
                .encode_post_convolution(&hidden, &coefficients, row_count, command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }
        if let Some(post_norm) = &self.post_norm {
            hidden = post_norm
                .encode(&hidden, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }
        self.connection
            .finish(started.residual, hidden, row_count, command_buffer)
            .map_err(TransformerLayerError::ResidualEncode)
    }

    fn encode(
        &self,
        stream: B::ScratchBuffer,
        row_count: u32,
        input_basis: InputBasis,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        body: impl FnOnce(
            B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<B::ScratchBuffer, TransformerLayerError<B>>,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let (started, hidden) = self.start(stream, row_count, input_basis, command_buffer)?;
        let update = body(hidden, command_buffer)?;
        self.finish(started, update, row_count, command_buffer)
    }
}

impl<B: Backend> TransformerLayer<B> {
    pub async fn new(
        model_dim: u32,
        hidden_dim: u32,
        num_layers: u32,
        layer_index: u32,
        data_type: DataType,
        layer_config: &TransformerLayerConfig,
        residual_config: &AnyResidualConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let prologue_params = parameter_tree.subtree("prologue");
        let prologue = try_join_all(layer_config.prologue_configs.iter().enumerate().map(|(i, cfg)| {
            StreamOp::new(
                model_dim,
                layer_index,
                num_layers,
                data_type,
                cfg,
                residual_config,
                prologue_params.subtree(&i.to_string()),
                context,
            )
        }))
        .await?
        .into_boxed_slice();

        let epilogue_params = parameter_tree.subtree("epilogue");
        let epilogue = try_join_all(layer_config.epilogue_configs.iter().enumerate().map(|(i, cfg)| {
            StreamOp::new(
                model_dim,
                layer_index,
                num_layers,
                data_type,
                cfg,
                residual_config,
                epilogue_params.subtree(&i.to_string()),
                context,
            )
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
        let mixer_supports_unrotated_input = mixer.supports_unrotated_input();
        let mixer_sublayer = Sublayer::new(
            model_dim,
            data_type,
            layer_index,
            "mixer",
            &layer_config.mixer.common,
            mixer_hadamard_factors,
            mixer_supports_unrotated_input,
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
        let mlp_supports_unrotated_input = mlp.supports_unrotated_input();
        let mlp_sublayer = Sublayer::new(
            model_dim,
            data_type,
            layer_index,
            "mlp",
            &layer_config.mlp.common,
            mlp_hadamard_factors,
            mlp_supports_unrotated_input,
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
        per_layer_inputs: Option<&B::ScratchBuffer>,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let command_buffer = &mut *command_buffer.debug_group(&format!("transformer layer {}", self.layer_index));
        let row_count = batch_dim.size();
        stream = self.encode_before_mlp(stream, per_layer_inputs, batch_dim, state, command_buffer)?;
        let pending = self.encode_until_mlp(stream, row_count, command_buffer)?;
        self.encode_from_mlp(pending, per_layer_inputs, batch_dim, command_buffer)
    }

    pub fn encode_before_mlp(
        &self,
        mut stream: B::ScratchBuffer,
        per_layer_inputs: Option<&B::ScratchBuffer>,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let row_count = batch_dim.size();
        let op_context = StreamOpContext {
            batch_dim,
            per_layer_inputs,
        };
        for op in &self.prologue {
            stream = op.encode(stream, &op_context, command_buffer)?;
        }
        let mixer_input_basis = self.mixer.input_basis(row_count);
        self.mixer_sublayer.encode(stream, row_count, mixer_input_basis, command_buffer, |hidden, command_buffer| {
            Ok(self.mixer.encode_for_basis(hidden, batch_dim, state, mixer_input_basis, command_buffer)?)
        })
    }

    pub fn encode_until_mlp<'a>(
        &'a self,
        stream: B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<PendingMlp<'a, B>, TransformerLayerError<B>> {
        let input_basis = self.mlp.input_basis(row_count);
        let (sublayer, hidden) = self.mlp_sublayer.start(stream, row_count, input_basis, command_buffer)?;
        let mlp = self.mlp.encode_start_for_basis(hidden, row_count, input_basis, command_buffer)?;
        Ok(PendingMlp {
            mlp,
            sublayer,
        })
    }

    pub fn encode_from_mlp(
        &self,
        pending: PendingMlp<'_, B>,
        per_layer_inputs: Option<&B::ScratchBuffer>,
        batch_dim: &BatchTopology<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let row_count = batch_dim.size();
        let update = pending.mlp.finish(command_buffer)?;
        let stream = self.mlp_sublayer.finish(pending.sublayer, update, row_count, command_buffer)?;
        let op_context = StreamOpContext {
            batch_dim,
            per_layer_inputs,
        };
        let mut stream = stream;
        for op in &self.epilogue {
            stream = op.encode(stream, &op_context, command_buffer)?;
        }
        Ok(stream)
    }
}
