use std::sync::Arc;

use derive_more::Debug;
use futures::future::try_join_all;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, kernel::ActivationTransform,
    },
    config::{
        residual::AnyResidualConfig,
        transformer::LoopConfig,
        transformer_layer::{SublayerConfigCommon, TransformerLayerConfig, TransformerLayerConvConfig},
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        block_attention::{ConnectionPlacement, SublayerRole},
        causal_conv::{CausalConv, CausalConvNewError, CausalConvState},
        convolution::{ConvolutionNewError, SeparableCausalConv},
        linear::{InputRhtParameters, Linear, LinearBlockError, LinearEncodeError},
        mixer::{Mixer, MixerEncodeError, MixerNewError, MixerState},
        mlp::{Mlp, MlpBlockNewError, MlpEncodeError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        residual::{ResidualConnection, ResidualEncodeError, ResidualNewError, SublayerUpdate},
        stream_op::{StreamOp, StreamOpContext, StreamOpEncodeError, StreamOpNewError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{maybe_mut::MaybeMut, maybe_owned::MaybeOwned, size_for_shape::size_for_shape},
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
    #[error("Causal convolution error: {0}")]
    CausalConvolution(#[from] CausalConvNewError<B>),
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
    output_conv: Option<CausalConv<B>>,
    post_norm: Option<Normalization<B>>,
    connection: ResidualConnection<B>,
    model_dim: u32,
    data_type: DataType,
    num_inputs: u32,
    separate_input_norms: bool,
}

/// The states of the stateful convolutions on a layer's sublayer outputs.
pub struct SublayerConvStates<B: Backend> {
    mixer: Option<CausalConvState<B>>,
    mlp: Option<CausalConvState<B>>,
}

impl<B: Backend> SublayerConvStates<B> {
    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        for state in [&mut self.mixer, &mut self.mlp].into_iter().flatten() {
            state.encode_accept(accepted_indices, command_buffer);
        }
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
        if let Some((transform, signs)) = &self.input_rht {
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
        placement: &ConnectionPlacement,
        name: &'static str,
        config: &SublayerConfigCommon,
        mut input_hadamard_factors: Option<InputRhtParameters<B>>,
        residual_config: &AnyResidualConfig,
        layer_parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let layer_index = placement.layer_index;
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
                Normalization::new_mixture(
                    config.separate_input_norms.then_some(config.num_inputs),
                    model_dim,
                    data_type,
                    data_type,
                    input_hadamard_factors.take(),
                    ShortcutMode::None,
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
                    norm_config,
                    &parameter_tree.subtree("post_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let output_conv = match &config.output_conv_config {
            Some(conv_config) => {
                Some(CausalConv::new(model_dim, data_type, conv_config, &parameter_tree.subtree("output_conv"), context).await?)
            },
            None => None,
        };

        let connection = ResidualConnection::new(
            residual_config,
            placement,
            model_dim,
            data_type,
            &parameter_tree.subtree("connection"),
            context,
        )
        .await?;

        Ok(Self {
            pre_norm,
            conv,
            output_conv,
            post_norm,
            connection,
            model_dim,
            data_type,
            num_inputs: config.num_inputs,
            separate_input_norms: config.separate_input_norms,
        })
    }

    fn create_output_conv_state(
        &self,
        context: &B::Context,
    ) -> Result<Option<CausalConvState<B>>, B::Error> {
        self.output_conv.as_ref().map(|conv| conv.create_state(context)).transpose()
    }

    /// Normalizes the `num_inputs` input-major inputs of `row_count` rows each.
    fn encode_pre_norm(
        &self,
        pre_norm: &Normalization<B>,
        input: &B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let command_buffer = &mut *command_buffer.debug_group("pre norm");
        let input_size = size_for_shape(&[row_count, self.model_dim], self.data_type);
        if !self.separate_input_norms {
            // Inputs that share one read arrive once.
            let input_count = (input.size() / input_size) as u32;
            return pre_norm.encode(input, 0, row_count * input_count, None::<&mut B::ScratchBuffer>, command_buffer);
        }
        let mut output = command_buffer.allocate_scratch(input_size * self.num_inputs as usize)?;
        for component in 0..self.num_inputs {
            let range = component as usize * input_size..(component as usize + 1) * input_size;
            pre_norm.encode_component(
                input.subrange(range.clone()),
                component,
                row_count,
                (&mut output).subrange_mut(range),
                command_buffer,
            );
        }
        Ok(output)
    }

    fn encode(
        &self,
        stream: B::ScratchBuffer,
        batch_dim: &BatchTopology<B>,
        has_messages: bool,
        output_conv_state: Option<&mut CausalConvState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        body: impl FnOnce(
            B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<B::ScratchBuffer, TransformerLayerError<B>>,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let row_count = batch_dim.size();
        self.connection.encode(stream, row_count, has_messages, command_buffer, |input, target, command_buffer| {
            let mut hidden = match (&self.pre_norm, input) {
                (Some(pre_norm), input) => self
                    .encode_pre_norm(pre_norm, &input, row_count, command_buffer)
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
                let (conv_output, coefficients) = conv.encode_pre_convolution(&hidden, row_count, command_buffer)?;
                let body_output = body(conv_output, command_buffer)?;
                hidden = conv
                    .encode_post_convolution(&body_output, &coefficients, row_count, command_buffer)
                    .map_err(TransformerLayerError::Backend)?;
            } else {
                hidden = body(hidden, command_buffer)?;
            }

            if let Some(output_conv) = &self.output_conv {
                let command_buffer = &mut *command_buffer.debug_group("output conv");
                // Without a post norm the convolution writes straight into the residual stream.
                if let (Some(target), None) = (target, &self.post_norm) {
                    output_conv
                        .encode_into(
                            &hidden,
                            self.model_dim,
                            None::<(&B::ScratchBuffer, u32)>,
                            batch_dim,
                            output_conv_state,
                            target.stream.subrange_mut(target.range),
                            target.accumulate,
                            command_buffer,
                        )
                        .map_err(TransformerLayerError::Backend)?;
                    return Ok(SublayerUpdate::Written);
                }
                hidden = output_conv
                    .encode(
                        &hidden,
                        self.model_dim,
                        None::<(&B::ScratchBuffer, u32)>,
                        batch_dim,
                        output_conv_state,
                        command_buffer,
                    )
                    .map_err(TransformerLayerError::Backend)?;
            }

            if let Some(post_norm) = &self.post_norm {
                hidden = post_norm
                    .encode(&hidden, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                    .map_err(TransformerLayerError::Backend)?;
            }

            Ok(SublayerUpdate::Returned(hidden))
        })
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
        loop_config: Option<&LoopConfig>,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let placement = |role: SublayerRole, config: &SublayerConfigCommon| ConnectionPlacement {
            layer_index,
            num_layers,
            role,
            num_inputs: config.num_inputs,
            num_reads: config.num_reads(),
            loop_config: loop_config.cloned(),
        };
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
        let mixer_sublayer = Sublayer::new(
            model_dim,
            data_type,
            &placement(SublayerRole::Mixer, &layer_config.mixer.common),
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
            &placement(SublayerRole::Mlp, &layer_config.mlp.common),
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

    /// The states of the stateful convolutions on the sublayer outputs, if the layer has any.
    pub fn create_conv_states(
        &self,
        context: &B::Context,
    ) -> Result<Option<SublayerConvStates<B>>, B::Error> {
        let mixer = self.mixer_sublayer.create_output_conv_state(context)?;
        let mlp = self.mlp_sublayer.create_output_conv_state(context)?;
        Ok((mixer.is_some() || mlp.is_some()).then_some(SublayerConvStates {
            mixer,
            mlp,
        }))
    }

    /// `has_messages` marks a repeated pass over looped layers.
    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        conv_states: Option<&mut SublayerConvStates<B>>,
        has_messages: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let command_buffer = &mut *command_buffer.debug_group(&format!("transformer layer {}", self.layer_index));
        let row_count = batch_dim.size();
        let op_context = StreamOpContext {
            batch_dim,
            per_layer_inputs,
        };

        for op in &self.prologue {
            stream = op.encode(stream, &op_context, command_buffer)?;
        }

        let (mixer_conv_state, mlp_conv_state) = match conv_states {
            Some(SublayerConvStates {
                mixer,
                mlp,
            }) => (mixer.as_mut(), mlp.as_mut()),
            None => (None, None),
        };

        stream = self.mixer_sublayer.encode(
            stream,
            batch_dim,
            has_messages,
            mixer_conv_state,
            command_buffer,
            |hidden, command_buffer| Ok(self.mixer.encode(hidden, batch_dim, state, command_buffer)?),
        )?;

        stream = self.mlp_sublayer.encode(
            stream,
            batch_dim,
            has_messages,
            mlp_conv_state,
            command_buffer,
            |hidden, command_buffer| Ok(self.mlp.encode(hidden, row_count, command_buffer)?),
        )?;

        for op in &self.epilogue {
            stream = op.encode(stream, &op_context, command_buffer)?;
        }

        Ok(stream)
    }
}
