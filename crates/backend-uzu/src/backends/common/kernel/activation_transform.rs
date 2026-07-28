use crate::backends::common::{
    Allocation, Backend, Encoder, Kernels,
    gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
    kernel::ActivationTransformKernel,
};

pub struct ActivationTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationTransformKernel,
    ops: ActivationTransformOp,
    group_size: u32,
}

impl<B: Backend> ActivationTransform<B> {
    pub fn new(
        context: &B::Context,
        data_type: crate::data_type::DataType,
        ops: ActivationTransformOp,
        group_size: u32,
    ) -> Result<Self, B::Error> {
        let ops = ops.validate();
        let kernel = <B::Kernels as Kernels>::ActivationTransformKernel::new(context, data_type, ops)?;
        Ok(Self {
            kernel,
            ops,
            group_size,
        })
    }

    pub fn input_rht(
        context: &B::Context,
        data_type: crate::data_type::DataType,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::INPUT_RHT, HADAMARD_TRANSFORM_BLOCK_SIZE as u32)
    }

    pub fn output_rht(
        context: &B::Context,
        data_type: crate::data_type::DataType,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::OUTPUT_RHT, HADAMARD_TRANSFORM_BLOCK_SIZE as u32)
    }

    pub fn quantize(
        context: &B::Context,
        data_type: crate::data_type::DataType,
        group_size: u32,
        emit_group_sums: bool,
    ) -> Result<Self, B::Error> {
        let ops = if emit_group_sums {
            ActivationTransformOp::INPUT_RHT | ActivationTransformOp::QUANTIZE | ActivationTransformOp::GROUP_SUMS
        } else {
            ActivationTransformOp::INPUT_RHT | ActivationTransformOp::QUANTIZE
        };
        Self::new(context, data_type, ops, group_size)
    }

    /// FP Hadamard (input- or output-order depending on construction).
    /// `input` and `output` must be distinct buffers. The two scratch buffers are
    /// placeholders for the quantized outputs this mode does not write.
    pub fn encode_fp(
        &self,
        input: &Allocation<B>,
        output: &mut Allocation<B>,
        q_scratch: &mut Allocation<B>,
        scales_scratch: &mut Allocation<B>,
        rht_factors: &Allocation<B>,
        element_count: u32,
        batch_size: u32,
        encoder: &mut Encoder<B>,
    ) {
        assert!(!self.ops.contains(ActivationTransformOp::QUANTIZE));
        self.kernel.encode(
            input,
            output,
            q_scratch,
            scales_scratch,
            None::<&mut Allocation<B>>,
            rht_factors,
            batch_size,
            element_count,
            self.group_size,
            encoder,
        );
    }

    /// Input RHT + symmetric int8 quantization.
    pub fn encode_quantize(
        &self,
        input: &Allocation<B>,
        fp_scratch: &mut Allocation<B>,
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
            fp_scratch,
            q_out,
            scales_out,
            group_sums_out,
            rht_factors,
            batch_size,
            element_count,
            self.group_size,
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
