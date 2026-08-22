use bitflags::bitflags;

use crate::{
    ClippingBounds,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{ActivationType, GatedActMulOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::GatedActMulKernel,
    },
    data_type::DataType,
};

#[repr(u32)]
#[derive(Clone, Copy)]
enum GatedActMulGroupSize {
    Size32 = 32,
    Size64 = 64,
    Size128 = 128,
}

impl GatedActMulGroupSize {
    fn from_u32(value: u32) -> Self {
        match value {
            32 => Self::Size32,
            64 => Self::Size64,
            128 => Self::Size128,
            _ => panic!("unsupported activation group size: {value}"),
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct GatedActMulOptions: u8 {
        const INTERLEAVED = 1 << 0;
        const HADAMARD = 1 << 1;
    }
}

/// Value transforms baked into a gated-activation kernel specialization.
#[derive(Debug, Clone, Copy, Default)]
pub struct GatedActMulSettings {
    pub activation_alpha: Option<f32>,
    pub gate_clipping: ClippingBounds,
    pub value_clipping: ClippingBounds,
}

pub struct GatedActMul<B: Backend> {
    kernel: <B::Kernels as Kernels>::GatedActMulKernel,
    ops: GatedActMulOp,
    options: GatedActMulOptions,
    settings: GatedActMulSettings,
    activation_group_size: u32,
    sum_group_size: u32,
}

impl<B: Backend> GatedActMul<B> {
    pub fn full_precision(
        context: &B::Context,
        data_type: DataType,
        interleaved: bool,
        use_hadamard: bool,
        settings: GatedActMulSettings,
    ) -> Result<Self, B::Error> {
        let mut options = GatedActMulOptions::empty();
        options.set(GatedActMulOptions::INTERLEAVED, interleaved);
        options.set(GatedActMulOptions::HADAMARD, use_hadamard);
        Self::new(
            context,
            data_type,
            GatedActMulOp::FullPrecision,
            options,
            HADAMARD_TRANSFORM_BLOCK_SIZE,
            HADAMARD_TRANSFORM_BLOCK_SIZE,
            settings,
        )
    }

    pub fn quantized(
        context: &B::Context,
        data_type: DataType,
        activation_group_size: u32,
        sum_group_size: Option<u32>,
        settings: GatedActMulSettings,
    ) -> Result<Self, B::Error> {
        let activation_group_size = GatedActMulGroupSize::from_u32(activation_group_size);
        let sum_group_size = sum_group_size.map(GatedActMulGroupSize::from_u32);
        Self::new(
            context,
            data_type,
            sum_group_size.map_or(GatedActMulOp::Quantize, |_| GatedActMulOp::QuantizeWithGroupSums),
            GatedActMulOptions::INTERLEAVED | GatedActMulOptions::HADAMARD,
            activation_group_size as u32,
            sum_group_size.unwrap_or(activation_group_size) as u32,
            settings,
        )
    }

    fn new(
        context: &B::Context,
        data_type: DataType,
        ops: GatedActMulOp,
        options: GatedActMulOptions,
        activation_group_size: u32,
        sum_group_size: u32,
        settings: GatedActMulSettings,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::GatedActMulKernel::new(
            context,
            data_type,
            ops,
            options.contains(GatedActMulOptions::INTERLEAVED),
            options.contains(GatedActMulOptions::HADAMARD),
            activation_group_size,
            sum_group_size,
            settings.activation_alpha.is_some(),
            settings.gate_clipping.min.is_some(),
            settings.gate_clipping.max.is_some(),
            settings.value_clipping.min.is_some(),
            settings.value_clipping.max.is_some(),
        )?;
        Ok(Self {
            kernel,
            ops,
            options,
            settings,
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
        assert_eq!(self.options.contains(GatedActMulOptions::INTERLEAVED), value_operand.is_none());
        assert_eq!(self.options.contains(GatedActMulOptions::HADAMARD), hadamard_factors.is_some());
        assert!(
            !self.options.contains(GatedActMulOptions::HADAMARD)
                || gated_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
        );
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
            self.settings.activation_alpha,
            self.settings.gate_clipping.min,
            self.settings.gate_clipping.max,
            self.settings.value_clipping.min,
            self.settings.value_clipping.max,
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
        assert!(self.options.contains(GatedActMulOptions::INTERLEAVED));
        assert!(self.options.contains(GatedActMulOptions::HADAMARD));
        assert!(gated_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));
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
            self.settings.activation_alpha,
            self.settings.gate_clipping.min,
            self.settings.gate_clipping.max,
            self.settings.value_clipping.min,
            self.settings.value_clipping.max,
            encoder,
        );
    }
}
