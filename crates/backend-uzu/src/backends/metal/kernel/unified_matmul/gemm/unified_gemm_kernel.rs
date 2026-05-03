use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            kernel::{BufferArg, BufferArgMut},
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{
                UnifiedGemmMetalKernel,
                unified_matmul::gemm::{GemmWeightsBuffers, UnifiedGemmSpecialization},
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
                specialization
                    .validate()
                    .map_err(|error| MetalError::CannotCreatePipelineState(format!("{error:?}")))?;
                let kernel = UnifiedGemmMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.tiling_config.simdgroups_m,
                    specialization.tiling_config.simdgroups_n,
                    specialization.input_prologue,
                    specialization.weights_storage.weight_prologue(),
                    specialization.compute,
                    specialization.output,
                    specialization.alignment,
                    specialization.weights_storage.bits_per_weight(),
                    specialization.weights_storage.group_size(),
                    specialization.weights_storage.use_mlx_quant(),
                    specialization.weights_storage.use_zero_points(),
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'activations, 'weights, 'result>(
        &mut self,
        context: &MetalContext,
        specialization: UnifiedGemmSpecialization,
        activations: impl BufferArg<'activations, <Metal as Backend>::DenseBuffer>,
        weights: GemmWeightsBuffers<'weights>,
        result: impl BufferArgMut<'result, <Metal as Backend>::DenseBuffer>,
        group_count_x: u32,
        group_count_y: u32,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let kernel = self.get_or_create(context, specialization)?;
        type Buf = <Metal as Backend>::DenseBuffer;
        let (weights_buf, scales, biases, zero_points): (&Buf, Option<&Buf>, Option<&Buf>, Option<&Buf>) = match weights
        {
            GemmWeightsBuffers::FullPrecision {
                weights,
            } => (weights, None, None, None),
            GemmWeightsBuffers::Mlx {
                weights,
                scales,
                biases,
            } => (weights, Some(scales), Some(biases), None),
            GemmWeightsBuffers::Awq {
                weights,
                scales,
                zero_points,
            } => (weights, Some(scales), None, Some(zero_points)),
        };
        kernel.encode(
            activations,
            weights_buf,
            result,
            scales,
            biases,
            zero_points,
            group_count_x,
            group_count_y,
            specialization.tiling_config,
            encoder,
        );
        Ok(())
    }
}
