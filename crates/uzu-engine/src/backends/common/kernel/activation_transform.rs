use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::{ActivationTransformKernel, matmul::Int8CodeLayout},
    },
    data_type::DataType,
};

fn assert_row_width(element_count: u32) {
    assert!(
        element_count.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE),
        "activation transform requires element_count ({element_count}) to be a multiple of {HADAMARD_TRANSFORM_BLOCK_SIZE}"
    );
}

pub const ACTIVATION_SCALE_GROUP_SIZE: u32 = 128;

#[derive(Clone, Copy)]
pub struct ActivationQuantization {
    scale_group_size: u32,
    sum_group_size: Option<u32>,
    code_layout: Int8CodeLayout,
}

impl ActivationQuantization {
    pub fn new(
        scale_group_size: u32,
        weight_group_size: u32,
        emit_group_sums: bool,
        code_layout: Int8CodeLayout,
    ) -> Option<Self> {
        (matches!(scale_group_size, 32 | 64 | 128)
            && matches!(weight_group_size, 32 | 64 | 128)
            && scale_group_size >= weight_group_size)
            .then_some(Self {
                scale_group_size,
                sum_group_size: emit_group_sums.then_some(weight_group_size),
                code_layout,
            })
    }

    pub const fn scale_group_size(self) -> u32 {
        self.scale_group_size
    }

    pub const fn sum_group_size(self) -> Option<u32> {
        self.sum_group_size
    }

    pub const fn code_layout(self) -> Int8CodeLayout {
        self.code_layout
    }
}

pub struct ActivationTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationTransformKernel,
    in_place: bool,
    has_bias: bool,
    quantization: Option<ActivationQuantization>,
}

impl<B: Backend> ActivationTransform<B> {
    fn new(
        context: &B::Context,
        data_type: DataType,
        bias_data_type: Option<DataType>,
        ops: ActivationTransformOp,
        in_place: bool,
        quantization: Option<ActivationQuantization>,
    ) -> Result<Self, B::Error> {
        let has_bias = bias_data_type.is_some();
        let (codes_grouped_by_nibble, scale_group_size, sum_group_size) = match quantization {
            Some(q) => (q.code_layout.is_grouped_by_nibble(), q.scale_group_size, q.sum_group_size),
            None => (false, HADAMARD_TRANSFORM_BLOCK_SIZE, None),
        };
        let kernel = <B::Kernels as Kernels>::ActivationTransformKernel::new(
            context,
            data_type,
            bias_data_type.unwrap_or(data_type),
            ops,
            codes_grouped_by_nibble,
            in_place,
            scale_group_size,
            sum_group_size.unwrap_or(HADAMARD_TRANSFORM_BLOCK_SIZE),
            has_bias,
        )?;
        Ok(Self {
            kernel,
            in_place,
            has_bias,
            quantization,
        })
    }

    pub fn input_rht(
        context: &B::Context,
        data_type: DataType,
        in_place: bool,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, None, ActivationTransformOp::InputRht, in_place, None)
    }

    pub fn output_rht(
        context: &B::Context,
        data_type: DataType,
        bias_data_type: Option<DataType>,
        in_place: bool,
    ) -> Result<Self, B::Error> {
        Self::new(context, data_type, bias_data_type, ActivationTransformOp::OutputRht, in_place, None)
    }

    pub fn quantize(
        context: &B::Context,
        data_type: DataType,
        quantization: ActivationQuantization,
    ) -> Result<Self, B::Error> {
        let op = quantization
            .sum_group_size()
            .map_or(ActivationTransformOp::Quantize, |_| ActivationTransformOp::QuantizeWithGroupSums);
        Self::new(context, data_type, None, op, false, Some(quantization))
    }

    pub fn scale_group_size(&self) -> u32 {
        self.quantization.expect("quantized activation transform required").scale_group_size()
    }

    pub fn sum_group_size(&self) -> Option<u32> {
        self.quantization.expect("quantized activation transform required").sum_group_size()
    }

    pub fn code_layout(&self) -> Int8CodeLayout {
        self.quantization.expect("quantized activation transform required").code_layout()
    }

    /// `input` and `output` must be distinct buffers.
    pub fn encode_fp(
        &self,
        input: impl BufferRef<Backend = B>,
        output: impl BufferMut<Backend = B>,
        rht_factors: impl BufferRef<Backend = B>,
        batch_size: u32,
        element_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        assert!(self.quantization.is_none() && !self.in_place && !self.has_bias);
        assert_row_width(element_count);
        self.kernel.encode(
            Some(input),
            Some(output),
            None::<&B::GlobalBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            rht_factors,
            batch_size,
            element_count,
            command_buffer,
        )?;
        Ok(())
    }

    pub fn encode_fp_in_place(
        &self,
        data: impl BufferMut<Backend = B>,
        rht_factors: impl BufferRef<Backend = B>,
        bias: Option<&B::GlobalBuffer>,
        batch_size: u32,
        element_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        assert!(self.quantization.is_none() && self.in_place && self.has_bias == bias.is_some());
        assert_row_width(element_count);
        self.kernel.encode(
            None::<&B::ScratchBuffer>,
            Some(data),
            bias,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            rht_factors,
            batch_size,
            element_count,
            command_buffer,
        )?;
        Ok(())
    }

    pub fn encode_quantize(
        &self,
        input: impl BufferRef<Backend = B>,
        q_out: impl BufferMut<Backend = B>,
        scales_out: impl BufferMut<Backend = B>,
        group_sums_out: Option<impl BufferMut<Backend = B>>,
        rht_factors: impl BufferRef<Backend = B>,
        batch_size: u32,
        element_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let quantization = self.quantization.expect("quantized activation transform required");
        assert_row_width(element_count);
        assert!(
            element_count.is_multiple_of(quantization.scale_group_size()),
            "quantized activation row ({element_count}) must be a multiple of scale group ({})",
            quantization.scale_group_size()
        );
        if let Some(group_size) = quantization.sum_group_size() {
            assert!(element_count.is_multiple_of(group_size));
        }
        self.kernel.encode(
            Some(input),
            None::<&mut B::ScratchBuffer>,
            None::<&B::GlobalBuffer>,
            Some(q_out),
            Some(scales_out),
            group_sums_out,
            rht_factors,
            batch_size,
            element_count,
            command_buffer,
        )?;
        Ok(())
    }
}
