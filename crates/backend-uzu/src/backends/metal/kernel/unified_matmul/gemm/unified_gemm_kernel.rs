use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::Encoder,
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{
                UnifiedGemmMetalKernel,
                unified_matmul::gemm::{GemmWeights, UnifiedGemmDispatch, UnifiedGemmSpecialization},
            },
        },
    },
};

pub(crate) struct UnifiedGemmKernel {
    data_type: DataType,
    kernels: HashMap<UnifiedGemmSpecialization, UnifiedGemmMetalKernel>,
}

impl UnifiedGemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        let _ = context;
        Ok(Self {
            data_type,
            kernels: HashMap::new(),
        })
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: UnifiedGemmSpecialization,
    ) -> Result<&UnifiedGemmMetalKernel, MetalError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = UnifiedGemmMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.tiling_config.simdgroups_m,
                    specialization.tiling_config.simdgroups_n,
                    specialization.input_prologue,
                    specialization.weight_prologue,
                    specialization.compute,
                    specialization.output_transform,
                    specialization.alignment,
                    specialization.bits_per_weight,
                    specialization.group_size,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode(
        &mut self,
        context: &MetalContext,
        dispatch: UnifiedGemmDispatch<'_>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let specialization = dispatch
            .specialization()
            .try_validate()
            .map_err(|error| MetalError::CannotCreatePipelineState(format!("{error:?}")))?;
        let kernel = self.get_or_create(context, specialization)?;
        let (weights, scales, biases, zero_points) = match &dispatch.weights {
            GemmWeights::FullPrecision {
                weights,
            } => (*weights, None, None, None),
            GemmWeights::Mlx {
                weights,
                scales,
                biases,
                ..
            } => (*weights, Some(*scales), Some(*biases), None),
            GemmWeights::Awq {
                weights,
                scales,
                zero_points,
                ..
            } => (*weights, Some(*scales), None, Some(*zero_points)),
        };
        kernel.encode(
            (dispatch.activations, dispatch.activations_offset),
            weights,
            dispatch.result,
            scales,
            biases,
            zero_points,
            dispatch.group_count_x,
            dispatch.group_count_y,
            dispatch.tiling_config,
            encoder,
        );
        Ok(())
    }
}
