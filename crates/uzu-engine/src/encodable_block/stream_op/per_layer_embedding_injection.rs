use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{GatedActMul, GatedActMulSettings, Kernels, TensorAddScaleKernel},
    },
    config::{activation::AnyActivation, stream_op::per_layer_embedding_injection::PerLayerEmbeddingInjectionConfig},
    data_type::DataType,
    encodable_block::{
        linear::Linear,
        normalization::{Normalization, ShortcutMode},
        stream_op::{StreamOpEncodeError, StreamOpNewError},
    },
    safetensors::ParameterTree,
};

pub struct PerLayerEmbeddingInjection<B: Backend> {
    gate: Box<dyn Linear<B>>,
    projection: Box<dyn Linear<B>>,
    norm: Normalization<B>,
    gate_act_mul: GatedActMul<B>,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    activation: AnyActivation,
    model_dim: u32,
    ple_dim: u32,
    layer_index: u32,
    num_layers: u32,
    data_type: DataType,
}

impl<B: Backend> PerLayerEmbeddingInjection<B> {
    pub async fn new(
        model_dim: u32,
        layer_index: u32,
        num_layers: u32,
        data_type: DataType,
        config: &PerLayerEmbeddingInjectionConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, StreamOpNewError<B>> {
        let gate =
            <dyn Linear<B>>::new(model_dim, config.ple_dim, data_type, false, &parameter_tree.subtree("gate"), context)
                .await?;
        let projection = <dyn Linear<B>>::new(
            config.ple_dim,
            model_dim,
            data_type,
            false,
            &parameter_tree.subtree("projection"),
            context,
        )
        .await?;
        let norm = Normalization::new(
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.norm_config,
            &parameter_tree.subtree("norm"),
            context,
        )
        .await?;

        let activation_alpha = config.activation.alpha();
        let settings = GatedActMulSettings {
            activation_alpha: (activation_alpha != 1.0).then_some(activation_alpha),
            ..Default::default()
        };
        let gate_act_mul = GatedActMul::full_precision(context, data_type, false, false, settings)
            .map_err(StreamOpNewError::Backend)?;
        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, true)
            .map_err(StreamOpNewError::Backend)?;

        Ok(Self {
            gate,
            projection,
            norm,
            gate_act_mul,
            residual_add,
            activation: config.activation.clone(),
            model_dim,
            ple_dim: config.ple_dim,
            layer_index,
            num_layers,
            data_type,
        })
    }

    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        per_layer_inputs: impl BufferRef<Backend = B>,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, StreamOpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("per layer embedding injection");

        let mut gate_input = command_buffer.allocate_scratch(stream.size()).map_err(StreamOpEncodeError::Backend)?;
        command_buffer.encode_copy(&stream, &mut gate_input);
        let gate_out = self.gate.encode(gate_input, row_count, command_buffer)?;

        let mut activated = command_buffer
            .allocate_scratch_for_shape(&[row_count, self.ple_dim], self.data_type)
            .map_err(StreamOpEncodeError::Backend)?;
        self.gate_act_mul.encode_fp(
            &gate_out,
            Some(per_layer_inputs),
            &mut activated,
            None::<&B::GlobalGpuBuffer>,
            self.ple_dim,
            row_count,
            self.layer_index * self.ple_dim,
            self.num_layers * self.ple_dim,
            self.activation.act_type(),
            command_buffer,
        );

        let projected = self.projection.encode(activated, row_count, command_buffer)?;
        let normed = self
            .norm
            .encode(&projected, 0, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(StreamOpEncodeError::Backend)?;

        let elements = row_count * self.model_dim;
        self.residual_add.encode(
            None::<&B::ScratchBuffer>,
            Some(&normed),
            &mut stream,
            elements,
            elements,
            1.0,
            command_buffer,
        );

        Ok(stream)
    }
}
