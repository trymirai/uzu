use std::collections::{HashMap, hash_map::Entry};

use super::spec::GemvSpecialization;
use crate::{
    DataType,
    backends::{
        common::kernel::matmul::MatmulError,
        metal::{Metal, context::MetalContext, kernel::MatmulGemvMetalKernel},
    },
};

pub(crate) struct GemvKernel {
    data_type: DataType,
    pipelines: HashMap<GemvSpecialization, MatmulGemvMetalKernel>,
}

impl GemvKernel {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let mut kernel = Self {
            data_type,
            pipelines: HashMap::new(),
        };
        for &config in GemvSpecialization::precompile_configs(data_type) {
            kernel.get_or_create(context, config)?;
        }
        Ok(kernel)
    }

    pub(crate) fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&MatmulGemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemvMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.threadgroup_rows,
                    specialization.threadgroup_cols,
                    specialization.threads_per_simdgroup_row,
                    specialization.threads_per_simdgroup_col,
                    specialization.elements_per_thread_row,
                    specialization.elements_per_thread_col,
                    specialization.is_accumulate,
                    specialization.is_bias,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }
}
