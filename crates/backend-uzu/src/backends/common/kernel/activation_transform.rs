use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::ActivationTransformKernel,
    },
    data_type::DataType,
};

fn assert_row_width(element_count: u32) {
    assert!(
        (element_count as usize).is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE),
        "activation transform requires element_count ({element_count}) to be a multiple of {HADAMARD_TRANSFORM_BLOCK_SIZE}"
    );
}

pub struct ActivationTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationTransformKernel,
    ops: ActivationTransformOp,
}

impl<B: Backend> ActivationTransform<B> {
    fn new(
        context: &B::Context,
        data_type: DataType,
        ops: ActivationTransformOp,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::ActivationTransformKernel::new(context, data_type, ops)?;
        Ok(Self {
            kernel,
            ops,
        })
    }

    pub fn input_rht(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::InputRht)
    }

    pub fn output_rht(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::OutputRht)
    }

    pub fn quantize(
        context: &B::Context,
        data_type: DataType,
        emit_group_sums: bool,
    ) -> Result<Self, B::Error> {
        let ops = if emit_group_sums {
            ActivationTransformOp::QuantizeWithGroupSums
        } else {
            ActivationTransformOp::Quantize
        };
        Self::new(context, data_type, ops)
    }

    /// `input` and `output` must be distinct buffers.
    pub fn encode_fp(
        &self,
        input: &Allocation<B>,
        output: &mut Allocation<B>,
        rht_factors: &Allocation<B>,
        batch_size: u32,
        element_count: u32,
        encoder: &mut Encoder<B>,
    ) {
        assert!(!self.quantizes());
        assert_row_width(element_count);
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
        assert!(self.quantizes());
        assert_row_width(element_count);
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

    fn quantizes(&self) -> bool {
        matches!(self.ops, ActivationTransformOp::Quantize | ActivationTransformOp::QuantizeWithGroupSums)
    }

    pub fn emit_group_sums(&self) -> bool {
        self.ops == ActivationTransformOp::QuantizeWithGroupSums
    }
}
