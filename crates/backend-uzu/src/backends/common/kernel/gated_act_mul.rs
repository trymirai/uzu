use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{ActivationType, GatedActMulOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::GatedActMulKernel,
    },
    data_type::DataType,
};

const SUPPORTED_GROUP_SIZES: [u32; 3] = [32, 64, 128];

fn assert_group_size(
    group_size: u32,
    name: &str,
) {
    assert!(SUPPORTED_GROUP_SIZES.contains(&group_size), "unsupported {name} ({group_size})");
}

fn assert_hadamard_width(
    gated_dim: u32,
    use_hadamard: bool,
) {
    assert!(!use_hadamard || gated_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE as u32));
}

pub struct GatedActMul<B: Backend> {
    kernel: <B::Kernels as Kernels>::GatedActMulKernel,
    ops: GatedActMulOp,
    interleaved: bool,
    use_hadamard: bool,
    activation_group_size: u32,
    sum_group_size: u32,
}

impl<B: Backend> GatedActMul<B> {
    pub fn full_precision(
        context: &B::Context,
        data_type: DataType,
        interleaved: bool,
        use_hadamard: bool,
    ) -> Result<Self, B::Error> {
        Self::new(
            context,
            data_type,
            GatedActMulOp::FullPrecision,
            interleaved,
            use_hadamard,
            HADAMARD_TRANSFORM_BLOCK_SIZE as u32,
            HADAMARD_TRANSFORM_BLOCK_SIZE as u32,
        )
    }

    pub fn quantized(
        context: &B::Context,
        data_type: DataType,
        activation_group_size: u32,
        sum_group_size: Option<u32>,
    ) -> Result<Self, B::Error> {
        assert_group_size(activation_group_size, "activation group");
        if let Some(group_size) = sum_group_size {
            assert_group_size(group_size, "correction group");
        }
        Self::new(
            context,
            data_type,
            sum_group_size.map_or(GatedActMulOp::Quantize, |_| GatedActMulOp::QuantizeWithGroupSums),
            true,
            true,
            activation_group_size,
            sum_group_size.unwrap_or(activation_group_size),
        )
    }

    fn new(
        context: &B::Context,
        data_type: DataType,
        ops: GatedActMulOp,
        interleaved: bool,
        use_hadamard: bool,
        activation_group_size: u32,
        sum_group_size: u32,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::GatedActMulKernel::new(
            context,
            data_type,
            ops,
            interleaved,
            use_hadamard,
            activation_group_size,
            sum_group_size,
        )?;
        Ok(Self {
            kernel,
            ops,
            interleaved,
            use_hadamard,
            activation_group_size,
            sum_group_size,
        })
    }

    pub fn encode_fp(
        &self,
        act_operand: &Allocation<B>,
        value_operand: Option<&Allocation<B>>,
        output: &mut Allocation<B>,
        hadamard_factors: Option<&Allocation<B>>,
        gated_dim: u32,
        batch_dim: u32,
        value_offset: u32,
        value_row_stride: u32,
        act_type: ActivationType,
        encoder: &mut Encoder<B>,
    ) {
        assert_eq!(self.ops, GatedActMulOp::FullPrecision);
        assert_eq!(self.interleaved, value_operand.is_none());
        assert_eq!(self.use_hadamard, hadamard_factors.is_some());
        assert_hadamard_width(gated_dim, self.use_hadamard);
        self.kernel.encode(
            act_operand,
            value_operand,
            Some(output),
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            hadamard_factors,
            gated_dim,
            batch_dim,
            value_offset,
            value_row_stride,
            act_type,
            encoder,
        );
    }

    pub fn encode_quantized(
        &self,
        act_operand: &Allocation<B>,
        values: &mut Allocation<B>,
        scales: &mut Allocation<B>,
        group_sums: Option<&mut Allocation<B>>,
        hadamard_factors: &Allocation<B>,
        gated_dim: u32,
        batch_dim: u32,
        act_type: ActivationType,
        encoder: &mut Encoder<B>,
    ) {
        assert!(matches!(self.ops, GatedActMulOp::Quantize | GatedActMulOp::QuantizeWithGroupSums));
        assert!(self.interleaved);
        assert!(self.use_hadamard);
        assert_hadamard_width(gated_dim, true);
        assert!(gated_dim.is_multiple_of(self.activation_group_size));
        assert_eq!(self.ops == GatedActMulOp::QuantizeWithGroupSums, group_sums.is_some());
        if self.ops == GatedActMulOp::QuantizeWithGroupSums {
            assert!(gated_dim.is_multiple_of(self.sum_group_size));
        }
        self.kernel.encode(
            act_operand,
            None::<&Allocation<B>>,
            None::<&mut Allocation<B>>,
            Some(values),
            Some(scales),
            group_sums,
            Some(hadamard_factors),
            gated_dim,
            batch_dim,
            0,
            0,
            act_type,
            encoder,
        );
    }
}
