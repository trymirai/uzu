use bitflags::bitflags;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::{ActivationType, GatedActMulOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::{ActivationQuantization, GatedActMulKernel},
    },
    config::clipping::ClippingBounds,
    data_type::DataType,
};

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
    options: GatedActMulOptions,
    settings: GatedActMulSettings,
    quantization: Option<ActivationQuantization>,
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
        Self::new(context, data_type, options, settings, None)
    }

    pub fn quantized(
        context: &B::Context,
        data_type: DataType,
        quantization: ActivationQuantization,
        settings: GatedActMulSettings,
    ) -> Result<Self, B::Error> {
        let options = GatedActMulOptions::INTERLEAVED | GatedActMulOptions::HADAMARD;
        Self::new(context, data_type, options, settings, Some(quantization))
    }

    fn new(
        context: &B::Context,
        data_type: DataType,
        options: GatedActMulOptions,
        settings: GatedActMulSettings,
        quantization: Option<ActivationQuantization>,
    ) -> Result<Self, B::Error> {
        let (ops, codes_grouped_by_nibble, scale_group_size, sum_group_size) = match quantization {
            Some(quantization) => (
                if quantization.sum_group_size().is_some() {
                    GatedActMulOp::QuantizeWithGroupSums
                } else {
                    GatedActMulOp::Quantize
                },
                quantization.code_layout().is_grouped_by_nibble(),
                quantization.scale_group_size(),
                quantization.sum_group_size().unwrap_or(quantization.scale_group_size()),
            ),
            None => (GatedActMulOp::FullPrecision, false, HADAMARD_TRANSFORM_BLOCK_SIZE, HADAMARD_TRANSFORM_BLOCK_SIZE),
        };
        let kernel = <B::Kernels as Kernels>::GatedActMulKernel::new(
            context,
            data_type,
            ops,
            codes_grouped_by_nibble,
            options.contains(GatedActMulOptions::INTERLEAVED),
            options.contains(GatedActMulOptions::HADAMARD),
            scale_group_size,
            sum_group_size,
            settings.activation_alpha.is_some(),
            settings.gate_clipping.into_pair().is_some(),
            settings.value_clipping.into_pair().is_some(),
        )?;
        Ok(Self {
            kernel,
            options,
            settings,
            quantization,
        })
    }

    pub fn encode_fp(
        &self,
        act_operand: impl BufferRef<Backend = B>,
        value_operand: Option<impl BufferRef<Backend = B>>,
        output: impl BufferMut<Backend = B>,
        hadamard_factors: Option<impl BufferRef<Backend = B>>,
        gated_dim: u32,
        batch_dim: u32,
        value_offset: u32,
        value_row_stride: u32,
        act_type: ActivationType,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        assert!(self.quantization.is_none());
        assert_eq!(self.options.contains(GatedActMulOptions::INTERLEAVED), value_operand.is_none());
        assert_eq!(self.options.contains(GatedActMulOptions::HADAMARD), hadamard_factors.is_some());
        assert!(
            !self.options.contains(GatedActMulOptions::HADAMARD)
                || gated_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
        );
        let (gate_clip_min, gate_clip_max) = self.settings.gate_clipping.into_pair().unzip();
        let (value_clip_min, value_clip_max) = self.settings.value_clipping.into_pair().unzip();
        self.kernel.encode(
            act_operand,
            value_operand,
            Some(output),
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            hadamard_factors,
            gated_dim,
            batch_dim,
            value_offset,
            value_row_stride,
            act_type,
            self.settings.activation_alpha,
            gate_clip_min,
            gate_clip_max,
            value_clip_min,
            value_clip_max,
            command_buffer,
        );
    }

    pub fn encode_quantized(
        &self,
        act_operand: impl BufferRef<Backend = B>,
        values: impl BufferMut<Backend = B>,
        scales: impl BufferMut<Backend = B>,
        group_sums: Option<impl BufferMut<Backend = B>>,
        hadamard_factors: impl BufferRef<Backend = B>,
        gated_dim: u32,
        batch_dim: u32,
        act_type: ActivationType,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let quantization = self.quantization.expect("quantized gated activation required");
        assert!(self.options.contains(GatedActMulOptions::INTERLEAVED));
        assert!(self.options.contains(GatedActMulOptions::HADAMARD));
        assert!(gated_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));
        assert!(gated_dim.is_multiple_of(quantization.scale_group_size()));
        assert_eq!(quantization.sum_group_size().is_some(), group_sums.is_some());
        if let Some(group_size) = quantization.sum_group_size() {
            assert!(gated_dim.is_multiple_of(group_size));
        }
        let (gate_clip_min, gate_clip_max) = self.settings.gate_clipping.into_pair().unzip();
        let (value_clip_min, value_clip_max) = self.settings.value_clipping.into_pair().unzip();
        self.kernel.encode(
            act_operand,
            None::<&B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
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
            gate_clip_min,
            gate_clip_max,
            value_clip_min,
            value_clip_max,
            command_buffer,
        );
    }
}
