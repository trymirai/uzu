use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::{activation_transform::ActivationTransformOp, hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE},
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
        (scale_group_size == ACTIVATION_SCALE_GROUP_SIZE && (!emit_group_sums || matches!(weight_group_size, 32 | 64)))
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

#[derive(Clone, Copy)]
pub enum ActivationEncoding {
    Int8(ActivationQuantization),
    PackedHalf {
        group_size: u32,
    },
}

pub struct ActivationTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationTransformKernel,
    in_place: bool,
    has_bias: bool,
    encoding: Option<ActivationEncoding>,
}

impl<B: Backend> ActivationTransform<B> {
    fn new(
        context: &B::Context,
        data_type: DataType,
        bias_data_type: Option<DataType>,
        ops: ActivationTransformOp,
        in_place: bool,
        encoding: Option<ActivationEncoding>,
    ) -> Result<Self, B::Error> {
        let has_bias = bias_data_type.is_some();
        let (codes_grouped_by_nibble, scale_group_size, sum_group_size) = match encoding {
            Some(ActivationEncoding::Int8(q)) => (
                q.code_layout().is_grouped_by_nibble(),
                q.scale_group_size(),
                q.sum_group_size().unwrap_or(q.scale_group_size()),
            ),
            Some(ActivationEncoding::PackedHalf {
                group_size,
            }) => (false, HADAMARD_TRANSFORM_BLOCK_SIZE, group_size),
            None => (false, HADAMARD_TRANSFORM_BLOCK_SIZE, HADAMARD_TRANSFORM_BLOCK_SIZE),
        };
        let kernel = <B::Kernels as Kernels>::ActivationTransformKernel::new(
            context,
            data_type,
            bias_data_type.unwrap_or(data_type),
            ops,
            codes_grouped_by_nibble,
            in_place,
            scale_group_size,
            sum_group_size,
            has_bias,
        )?;
        Ok(Self {
            kernel,
            in_place,
            has_bias,
            encoding,
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
        encoding: ActivationEncoding,
    ) -> Result<Self, B::Error> {
        let op = match encoding {
            ActivationEncoding::Int8(q) if q.sum_group_size().is_some() => ActivationTransformOp::QuantizeWithGroupSums,
            ActivationEncoding::Int8(_) => ActivationTransformOp::Quantize,
            ActivationEncoding::PackedHalf {
                ..
            } => ActivationTransformOp::PackHalf,
        };
        Self::new(context, data_type, None, op, false, Some(encoding))
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
    ) {
        assert!(self.encoding.is_none() && !self.in_place && !self.has_bias);
        assert_row_width(element_count);
        self.kernel.encode(
            Some(input),
            Some(output),
            None::<&B::GlobalGpuBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            rht_factors,
            batch_size,
            element_count,
            None::<&mut B::ScratchBuffer>,
            command_buffer,
        );
    }

    pub fn encode_fp_in_place(
        &self,
        data: impl BufferMut<Backend = B>,
        rht_factors: impl BufferRef<Backend = B>,
        bias: Option<&B::GlobalGpuBuffer>,
        batch_size: u32,
        element_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        assert!(self.encoding.is_none() && self.in_place && self.has_bias == bias.is_some());
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
            None::<&mut B::ScratchBuffer>,
            command_buffer,
        );
    }

    pub fn encode_quantize(
        &self,
        input: impl BufferRef<Backend = B>,
        values: impl BufferMut<Backend = B>,
        scales: impl BufferMut<Backend = B>,
        group_sums: Option<impl BufferMut<Backend = B>>,
        rht_factors: impl BufferRef<Backend = B>,
        batch_size: u32,
        element_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let Some(ActivationEncoding::Int8(q)) = self.encoding else {
            panic!("INT8 activation transform required")
        };
        assert_row_width(element_count);
        assert!(element_count.is_multiple_of(q.scale_group_size()));
        assert_eq!(q.sum_group_size().is_some(), group_sums.is_some());
        if let Some(group_size) = q.sum_group_size() {
            assert!(element_count.is_multiple_of(group_size));
        }
        self.kernel.encode(
            Some(input),
            None::<&mut B::ScratchBuffer>,
            None::<&B::GlobalGpuBuffer>,
            Some(values),
            Some(scales),
            group_sums,
            rht_factors,
            batch_size,
            element_count,
            None::<&mut B::ScratchBuffer>,
            command_buffer,
        );
    }

    pub fn encode_packed_half(
        &self,
        input: impl BufferRef<Backend = B>,
        output: impl BufferMut<Backend = B>,
        rht_factors: impl BufferRef<Backend = B>,
        batch_size: u32,
        element_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let Some(ActivationEncoding::PackedHalf {
            group_size,
        }) = self.encoding
        else {
            panic!("packed-half activation transform required")
        };
        assert_row_width(element_count);
        assert!(element_count.is_multiple_of(group_size) && batch_size <= 8);
        self.kernel.encode(
            Some(input),
            None::<&mut B::ScratchBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            rht_factors,
            batch_size,
            element_count,
            Some(output),
            command_buffer,
        );
    }
}
