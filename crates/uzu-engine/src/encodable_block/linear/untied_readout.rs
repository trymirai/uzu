use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::matmul::{ActivationFormat, MatmulA},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        LinearEncodeError, LinearInputPreparation, LinearMatmul, LinearMatmulError, input_rht::InputRht,
    },
    safetensors::ParameterTree,
};

pub struct UntiedReadout<B: Backend> {
    linear: LinearMatmul<B>,
    input_rht: Option<InputRht<B>>,
}

impl<B: Backend> UntiedReadout<B> {
    pub fn load(
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
        spec: AnyWeightMatrixSpec,
        tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, LinearMatmulError<B>> {
        let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode: IncoherenceProcessingMode::Input,
            ..
        }) = spec
        else {
            return Ok(Self {
                linear: LinearMatmul::load(
                    model_dim, vocab_size, data_type, data_type, data_type, spec, None, tree, None, context,
                )?,
                input_rht: None,
            });
        };

        let mut linear = LinearMatmul::load(
            model_dim,
            vocab_size,
            data_type,
            data_type,
            data_type,
            *quantization_spec,
            None,
            &tree.subtree("quantized"),
            None,
            context,
        )?;
        let rht_signs = tree.subtree("incoherence_signs").read_buffer("input_signs", &[model_dim], DataType::I32)?;
        let preparation = LinearInputPreparation {
            rht_signs,
            activation_quantization: linear.prepare_a8(context),
        };
        let input_rht = InputRht::new(data_type, preparation, /* in_place */ false, context)
            .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            linear,
            input_rht: Some(input_rht),
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        output_dim: u32,
        gather_indices: Option<impl BufferRef<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let Some(input_rht) = &self.input_rht else {
            let a = MatmulA::FullPrecision {
                values: input,
                offset: 0,
            };
            return self.linear.encode_readout(a, batch_dim, output_dim, gather_indices, command_buffer);
        };

        let format = if gather_indices.is_some() {
            ActivationFormat::Bf16
        } else {
            self.linear.select_readout_activation_format(batch_dim, output_dim, command_buffer.context())
        };
        let input = input_rht.prepare(input, batch_dim, format, command_buffer).map_err(LinearEncodeError::Backend)?;
        self.linear.encode_readout(input.as_matmul_a(), batch_dim, output_dim, gather_indices, command_buffer)
    }
}
