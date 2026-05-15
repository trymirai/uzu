use std::collections::{HashMap, hash_map::Entry};

use super::dispatch::{GemmDispatch, GemmSpecialization, GemmWeights};
use crate::{
    DataType,
    backends::{
        common::Encoder,
        metal::{Metal, context::MetalContext, error::MetalError, kernel::GemmMetalKernel},
    },
};

pub(crate) struct GemmKernel {
    data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
}

impl GemmKernel {
    pub(crate) fn new(
        _context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            data_type,
            kernels: HashMap::new(),
        })
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
                    specialization.tiling_config.threadgroup_m,
                    specialization.tiling_config.threadgroup_n,
                    specialization.tiling_config.threadgroup_k,
                    specialization.tiling_config.simdgroups_m,
                    specialization.tiling_config.simdgroups_n,
                    specialization.transpose_b,
                    specialization.input_prologue,
                    specialization.weight_prologue,
                    specialization.compute,
                    specialization.output_transform,
                    specialization.alignment.bits(),
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
        dispatch: GemmDispatch<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let specialization = dispatch
            .specialization()
            .try_validate()
            .map_err(|error| MetalError::CannotCreatePipelineState(format!("{error:?}")))?;
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
            } => (*weights, Some(*scales), Some(*biases), None),
            GemmWeights::ScaleZeroPoint {
                weights,
                scales,
                zero_points,
                ..
            } => (*weights, Some(*scales), None, Some(*zero_points)),
        };
        kernel.encode(
            (dispatch.a, dispatch.a_offset),
            (b, dispatch.b_offset),
            dispatch.d,
            scales,
            biases,
            zero_points,
            std::slice::from_ref(&dispatch.params),
            dispatch.group_count_x,
            dispatch.group_count_y,
            encoder,
        );
        Ok(())
    }
}
