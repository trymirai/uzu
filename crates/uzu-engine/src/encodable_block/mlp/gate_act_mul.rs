use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::ActivationType,
        kernel::{
            GatedActMul, GatedActMulSettings,
            matmul::{A8ActivationPlan, ActivationFormat},
        },
    },
    config::{activation::AnyActivation, clipping::ClippingBounds},
    data_type::DataType,
    encodable_block::linear::{LinearInput, LinearInputPreparation},
};

pub struct MlpGateActMulEncodable<B: Backend> {
    fp_kernel: GatedActMul<B>,
    activation: AnyActivation,
    hidden_dim: u32,
    data_type: DataType,
    hadamard_factors: Option<Allocation<B>>,
    a8_plan: Option<A8ActivationPlan>,
    quantized_kernel: Option<GatedActMul<B>>,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: AnyActivation,
        gate_clipping: ClippingBounds,
        value_clipping: ClippingBounds,
        hidden_dim: u32,
        input_preparation: Option<LinearInputPreparation<B>>,
    ) -> Result<Self, B::Error> {
        let (hadamard_factors, a8_plan) = input_preparation
            .map_or((None, None), |preparation| (Some(preparation.input_factors), preparation.a8_plan));
        let settings = GatedActMulSettings {
            activation_alpha: activation.custom_alpha(),
            gate_clipping,
            value_clipping,
        };
        let fp_kernel = GatedActMul::full_precision(context, data_type, true, hadamard_factors.is_some(), settings)?;
        let quantized_kernel = a8_plan
            .map(|plan| {
                GatedActMul::quantized(context, data_type, plan.activation_group_size, plan.sum_group_size, settings)
            })
            .transpose()?;
        Ok(Self {
            fp_kernel,
            activation,
            hidden_dim,
            data_type,
            hadamard_factors,
            a8_plan,
            quantized_kernel,
        })
    }

    pub fn encode_for_linear(
        &self,
        encoder: &mut Encoder<B>,
        fused_up: &Allocation<B>,
        batch_dim: u32,
        act_format: ActivationFormat,
    ) -> Result<LinearInput<B>, B::Error> {
        encoder.push_debug_group("gate act mul");

        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        let input = if act_format == ActivationFormat::Int8
            && let Some(plan) = self.a8_plan
        {
            let kernel = self.quantized_kernel.as_ref().expect("INT8 input requires a quantized gate kernel");
            let mut values = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(
                &[batch_dim, self.hidden_dim.div_ceil(plan.activation_group_size)],
                DataType::F32,
            ))?;
            let mut group_sums = plan
                .sum_group_size
                .map(|group_size| {
                    encoder.allocate_scratch(size_for_shape(
                        &[batch_dim, self.hidden_dim.div_ceil(group_size)],
                        DataType::I32,
                    ))
                })
                .transpose()?;
            kernel.encode_quantized(
                fused_up,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                self.hadamard_factors.as_ref().expect("INT8 input requires RHT factors"),
                self.hidden_dim,
                batch_dim,
                self.activation.act_type(),
                encoder,
            );
            LinearInput::Int8Symmetric {
                values,
                scales,
                group_sums,
                group_size: plan.activation_group_size,
            }
        } else {
            let mut hidden = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], self.data_type))?;
            self.fp_kernel.encode_fp(
                fused_up,
                None,
                &mut hidden,
                self.hadamard_factors.as_ref(),
                self.hidden_dim,
                batch_dim,
                0,
                0,
                self.activation.act_type(),
                encoder,
            );
            LinearInput::FullPrecision(hidden)
        };

        encoder.pop_debug_group();

        Ok(input)
    }
}
