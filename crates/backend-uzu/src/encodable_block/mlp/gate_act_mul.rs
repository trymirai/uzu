use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{ActivationType, GatedActMulOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::{
            GatedActMulKernel,
            matmul::{A8ActivationPlan, ActivationFormat},
        },
    },
    config::activation::AnyActivation,
    data_type::DataType,
    encodable_block::linear::{LinearInput, LinearInputPreparation},
};

pub struct MlpGateActMulEncodable<B: Backend> {
    fp_kernel: <B::Kernels as Kernels>::GatedActMulKernel,
    activation: AnyActivation,
    hidden_dim: u32,
    data_type: DataType,
    hadamard_factors: Option<Allocation<B>>,
    a8_plan: Option<A8ActivationPlan>,
    quantized_kernel: Option<<B::Kernels as Kernels>::GatedActMulKernel>,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: AnyActivation,
        hidden_dim: usize,
        input_preparation: Option<LinearInputPreparation<B>>,
    ) -> Result<Self, B::Error> {
        let (hadamard_factors, a8_plan) = input_preparation
            .map(|preparation| (Some(preparation.input_factors), preparation.a8_plan))
            .unwrap_or((None, None));
        let fp_kernel = <B::Kernels as Kernels>::GatedActMulKernel::new(
            context,
            data_type,
            GatedActMulOp::FullPrecision,
            true,
            hadamard_factors.is_some(),
            HADAMARD_TRANSFORM_BLOCK_SIZE as u32,
            HADAMARD_TRANSFORM_BLOCK_SIZE as u32,
        )?;
        let quantized_kernel = a8_plan
            .map(|plan| {
                <B::Kernels as Kernels>::GatedActMulKernel::new(
                    context,
                    data_type,
                    if plan.sum_group_size.is_some() {
                        GatedActMulOp::QuantizeWithGroupSums
                    } else {
                        GatedActMulOp::Quantize
                    },
                    true,
                    true,
                    plan.activation_group_size,
                    plan.sum_group_size.unwrap_or(plan.activation_group_size),
                )
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
        batch_dim: usize,
        format: ActivationFormat,
    ) -> Result<LinearInput<B>, B::Error> {
        encoder.push_debug_group("gate act mul");

        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        let input = if format == ActivationFormat::Int8 && self.a8_plan.is_some() {
            let plan = self.a8_plan.expect("INT8 input requires an A8 plan");
            let kernel = self.quantized_kernel.as_ref().expect("INT8 input requires a quantized gate kernel");
            let mut values = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(
                &[batch_dim, self.hidden_dim / plan.activation_group_size as usize],
                DataType::F32,
            ))?;
            let mut group_sums = plan
                .sum_group_size
                .map(|group_size| {
                    encoder.allocate_scratch(size_for_shape(
                        &[batch_dim, self.hidden_dim / group_size as usize],
                        DataType::I32,
                    ))
                })
                .transpose()?;
            kernel.encode(
                fused_up,
                None::<&Allocation<B>>,
                None::<&mut Allocation<B>>,
                Some(&mut values),
                Some(&mut scales),
                group_sums.as_mut(),
                Some(self.hadamard_factors.as_ref().expect("INT8 input requires RHT factors")),
                self.hidden_dim as u32,
                batch_dim as u32,
                0,
                0,
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
            self.fp_kernel.encode(
                fused_up,
                None::<&Allocation<B>>,
                Some(&mut hidden),
                None::<&mut Allocation<B>>,
                None::<&mut Allocation<B>>,
                None::<&mut Allocation<B>>,
                self.hadamard_factors.as_ref(),
                self.hidden_dim as u32,
                batch_dim as u32,
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
