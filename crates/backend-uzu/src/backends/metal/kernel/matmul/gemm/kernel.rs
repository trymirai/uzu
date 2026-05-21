use std::collections::{HashMap, hash_map::Entry};

use super::{dispatch::GemmDispatch, specialization::GemmSpecialization, weights::GemmWeights};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::gemm::GemmWeightPrologueKind,
            kernel::matmul::MatmulArguments,
        },
        metal::{
            Metal, context::MetalContext, error::MetalError, kernel::GemmMetalKernel,
            metal_extensions::DeviceExt,
        },
    },
};

pub(crate) struct GemmKernel {
    data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
    mxu_eligible: bool,
}

impl GemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        let mxu_eligible =
            context.device.supports_mxu() && matches!(data_type, DataType::F16 | DataType::BF16);
        let mut kernel = Self {
            data_type,
            kernels: HashMap::new(),
            mxu_eligible,
        };
        for specialization in GemmSpecialization::precompile_configs(data_type) {
            kernel.get_or_create(context, specialization)?;
        }
        Ok(kernel)
    }

    /// Whether this kernel will route FP gemms through the MXU path.
    pub(crate) fn uses_mxu(&self) -> bool {
        self.mxu_eligible
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemmSpecialization,
    ) -> Result<&GemmMetalKernel, MetalError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.weight_prologue,
                    specialization.bits_per_weight,
                    specialization.group_size,
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    /// Unified entry point — encodes both FP and quantized GEMMs.
    pub(crate) fn encode<'a>(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let dispatch = GemmDispatch::from_arguments(arguments, self.data_type, self.mxu_eligible);
        let specialization = dispatch.specialization();
        specialization.validate().map_err(MetalError::InvalidGemmSpecialization)?;
        let kernel = self.get_or_create(context, specialization)?;
        let (b, scales, biases, zero_points) = match &dispatch.b {
            GemmWeights::FullPrecision {
                weights,
            } => (*weights, None, None, None),
            GemmWeights::ScaleBias {
                weights,
                scales,
                biases,
                ..
            } => {
                debug_assert_eq!(specialization.weight_prologue, GemmWeightPrologueKind::ScaleBiasDequant);
                (*weights, Some(*scales), Some(*biases), None)
            },
            GemmWeights::ScaleZeroPoint {
                weights,
                scales,
                zero_points,
                ..
            } => {
                debug_assert_eq!(specialization.weight_prologue, GemmWeightPrologueKind::ScaleZeroPointDequant);
                (*weights, Some(*scales), None, Some(*zero_points))
            },
        };
        kernel.encode(
            (dispatch.a, dispatch.a_offset),
            (b, dispatch.b_offset),
            dispatch.d,
            scales,
            biases,
            zero_points,
            dispatch.output_bias,
            std::slice::from_ref(&dispatch.params),
            dispatch.group_count_x,
            dispatch.group_count_y,
            encoder,
        );
        Ok(())
    }
}
