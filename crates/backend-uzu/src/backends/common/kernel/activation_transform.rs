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

pub const ACTIVATION_SCALE_GROUP_SIZE: u32 = 128;

pub struct ActivationTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationTransformKernel,
    ops: ActivationTransformOp,
    in_place: bool,
    activation_group_size: usize,
    sum_group_size: Option<u32>,
}

impl<B: Backend> ActivationTransform<B> {
    fn new(
        context: &B::Context,
        data_type: DataType,
        ops: ActivationTransformOp,
        in_place: bool,
        activation_group_size: usize,
        sum_group_size: Option<u32>,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::ActivationTransformKernel::new(
            context,
            data_type,
            ops,
            in_place,
            activation_group_size as u32,
            sum_group_size.unwrap_or(HADAMARD_TRANSFORM_BLOCK_SIZE as u32),
        )?;
        Ok(Self {
            kernel,
            ops,
            in_place,
            activation_group_size,
            sum_group_size,
        })
    }

    pub fn input_rht(
        context: &B::Context,
        data_type: DataType,
        in_place: bool,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::InputRht, in_place, HADAMARD_TRANSFORM_BLOCK_SIZE, None)
    }

    pub fn output_rht(
        context: &B::Context,
        data_type: DataType,
        in_place: bool,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, ActivationTransformOp::OutputRht, in_place, HADAMARD_TRANSFORM_BLOCK_SIZE, None)
    }

    pub fn quantize(
        context: &B::Context,
        data_type: DataType,
        activation_group_size: usize,
        sum_group_size: Option<u32>,
    ) -> Result<Self, B::Error> {
        let ops = if sum_group_size.is_some() {
            ActivationTransformOp::QuantizeWithGroupSums
        } else {
            ActivationTransformOp::Quantize
        };
        if let Some(group_size) = sum_group_size {
            assert!(matches!(group_size, 32 | 64 | 128), "unsupported correction group ({group_size})");
        }
        assert!(
            matches!(activation_group_size, 32 | 64 | 128),
            "unsupported activation group ({activation_group_size})"
        );
        Self::new(context, data_type, ops, false, activation_group_size, sum_group_size)
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
        assert!(!self.quantizes() && !self.in_place);
        assert_row_width(element_count);
        self.kernel.encode(
            Some(input),
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

    pub fn encode_fp_in_place(
        &self,
        data: &mut Allocation<B>,
        rht_factors: &Allocation<B>,
        batch_size: u32,
        element_count: u32,
        encoder: &mut Encoder<B>,
    ) {
        assert!(!self.quantizes() && self.in_place);
        assert_row_width(element_count);
        self.kernel.encode(
            None::<&Allocation<B>>,
            Some(data),
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
        assert!(
            element_count.is_multiple_of(self.activation_group_size as u32),
            "quantized activation row ({element_count}) must be a multiple of scale group ({})",
            self.activation_group_size
        );
        if let Some(group_size) = self.sum_group_size {
            assert!(element_count.is_multiple_of(group_size));
        }
        self.kernel.encode(
            Some(input),
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
        self.sum_group_size.is_some()
    }

    pub fn activation_group_size(&self) -> usize {
        self.activation_group_size
    }

    pub fn sum_group_size(&self) -> Option<u32> {
        self.sum_group_size
    }
}
