use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            gpu_types::unified_gemm::{GemmFragmentTile, GemmSimdgroupTile, GemmThreadgroupTile},
            kernel::{BufferArg, BufferArgMut},
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{UnifiedGemmMetalKernel, unified_matmul::gemm::UnifiedGemmSpecialization},
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
                    specialization.tile.simdgroups_m,
                    specialization.tile.simdgroups_n,
                    specialization.input_prologue,
                    specialization.weights_storage.weight_prologue(),
                    specialization.compute,
                    specialization.output,
                    specialization.alignment,
                    specialization.weights_storage.bits_per_weight(),
                    specialization.weights_storage.group_size(),
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, 'b, 'd>(
        &mut self,
        context: &MetalContext,
        specialization: UnifiedGemmSpecialization,
        a: impl BufferArg<'a, <Metal as Backend>::Buffer>,
        b: impl BufferArg<'b, <Metal as Backend>::Buffer>,
        d: impl BufferArgMut<'d, <Metal as Backend>::Buffer>,
        group_count_x: u32,
        group_count_y: u32,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let kernel = self.get_or_create(context, specialization)?;
        kernel.encode(
            a,
            b,
            d,
            group_count_x,
            group_count_y,
            GemmThreadgroupTile {
                m: specialization.tile.threadgroup_m,
                n: specialization.tile.threadgroup_n,
                k: specialization.tile.threadgroup_k,
            },
            GemmSimdgroupTile {
                m: specialization.tile.simdgroup_m,
                n: specialization.tile.simdgroup_n,
                k: specialization.tile.simdgroup_k,
            },
            GemmFragmentTile {
                m: specialization.tile.fragment_m,
                n: specialization.tile.fragment_n,
                k: specialization.tile.fragment_k,
            },
            encoder,
        );
        Ok(())
    }
}
