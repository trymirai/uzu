use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::matmul::{ActivationFormat, MatmulA},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearInputPreparation, LinearMatmul, LinearMatmulError, input_rht::InputRht,
    },
    parameters::ParameterTree,
};

pub struct UntiedReadout<B: Backend> {
    linear: LinearMatmul<B>,
    input_rht: Option<InputRht<B>>,
}

impl<B: Backend> UntiedReadout<B> {
    pub fn load(
        context: &B::Context,
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
    ) -> Result<Self, LinearMatmulError<B>> {
        let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            adapter_spec: None,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode: IncoherenceProcessingMode::Input,
            ..
        }) = spec
        else {
            return Ok(Self {
                linear: LinearMatmul::load(
                    context, spec, model_dim, vocab_size, data_type, data_type, data_type, tree, None, None,
                )?,
                input_rht: None,
            });
        };

        let mut linear = LinearMatmul::load(
            context,
            *quantization_spec,
            model_dim,
            vocab_size,
            data_type,
            data_type,
            data_type,
            &tree.subtree("quantized"),
            None,
            None,
        )?;
        let rht_signs = tree
            .subtree("incoherence_signs")
            .leaf("input_signs")?
            .validate(&[model_dim], DataType::I32)?
            .read_allocation()?;
        let preparation = LinearInputPreparation {
            rht_signs,
            a8_plan: linear.prepare_a8(context),
        };
        let input_rht = InputRht::new(context, data_type, preparation, /* in_place */ false)
            .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            linear,
            input_rht: Some(input_rht),
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        batch_dim: u32,
        gather: Option<Gather<'_, B>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let Some(input_rht) = &self.input_rht else {
            let a = MatmulA::FullPrecision {
                values: input,
                offset: 0,
            };
            return self.linear.encode_with_a(a, batch_dim, gather, encoder);
        };

        let format = if gather.is_some() {
            ActivationFormat::Bf16
        } else {
            self.linear.select_activation_format(batch_dim, encoder.context())
        };
        let input = input_rht.prepare(input, batch_dim, format, encoder)?;
        self.linear.encode_with_a(input.as_matmul_a(), batch_dim, gather, encoder)
    }
}
