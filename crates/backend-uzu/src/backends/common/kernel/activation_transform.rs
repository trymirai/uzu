use crate::backends::common::{
    Allocation, Backend, Encoder, Kernels, gpu_types::ActivationTransformOp, kernel::ActivationTransformKernel,
};

pub struct ActivationTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationTransformKernel,
    ops: ActivationTransformOp,
}

impl<B: Backend> ActivationTransform<B> {
    pub fn new(
        context: &B::Context,
        data_type: crate::data_type::DataType,
        ops: ActivationTransformOp,
    ) -> Result<Self, B::Error> {
        let ops = ops.validate();
        let kernel = <B::Kernels as Kernels>::ActivationTransformKernel::new(context, data_type, ops)?;
        Ok(Self {
            kernel,
            ops,
        })
    }

    pub fn input_rht(
        context: &B::Context,
        data_type: crate::data_type::DataType,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::INPUT_RHT)
    }

    pub fn output_rht(
        context: &B::Context,
        data_type: crate::data_type::DataType,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::OUTPUT_RHT)
    }

    pub fn quantize(
        context: &B::Context,
        data_type: crate::data_type::DataType,
        emit_group_sums: bool,
    ) -> Result<Self, B::Error> {
        let ops = if emit_group_sums {
            ActivationTransformOp::INPUT_RHT | ActivationTransformOp::QUANTIZE | ActivationTransformOp::GROUP_SUMS
        } else {
            ActivationTransformOp::INPUT_RHT | ActivationTransformOp::QUANTIZE
        };
        Self::new(context, data_type, ops)
    }

    /// FP Hadamard (input- or output-order depending on construction).
    /// `input` and `output` must be distinct buffers.
    pub fn encode_fp(
        &self,
        input: &Allocation<B>,
        output: &mut Allocation<B>,
        rht_factors: &Allocation<B>,
        element_count: u32,
        batch_size: u32,
        encoder: &mut Encoder<B>,
    ) {
        assert!(!self.ops.contains(ActivationTransformOp::QUANTIZE));
        self.kernel.encode(
            input,
            Some(output),
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            rht_factors,
            batch_size,
            element_count,
            encoder,
        );
    }

    /// Input RHT + symmetric int8 quantization.
    pub fn encode_quantize(
        &self,
        input: &Allocation<B>,
        q_out: &mut Allocation<B>,
        scales_out: &mut Allocation<B>,
        group_sums_out: Option<&mut Allocation<B>>,
        rht_factors: &Allocation<B>,
        batch_size: u32,
        element_count: u32,
        encoder: &mut Encoder<B>,
    ) {
        assert!(self.ops.contains(ActivationTransformOp::QUANTIZE));
        self.kernel.encode(
            input,
            None::<&mut Allocation<B>>,
            Some(q_out),
            Some(scales_out),
            group_sums_out,
            rht_factors,
            batch_size,
            element_count,
            encoder,
        );
    }

    pub fn ops(&self) -> ActivationTransformOp {
        self.ops
    }

    pub fn emit_group_sums(&self) -> bool {
        self.ops.contains(ActivationTransformOp::GROUP_SUMS)
    }
}
