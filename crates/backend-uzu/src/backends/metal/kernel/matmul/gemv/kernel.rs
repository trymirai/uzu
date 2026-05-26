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
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, MatmulGemvMetalKernel>,
}

impl GemvKernel {
    pub(crate) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let mut kernel = Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        };
        for &config in GemvSpecialization::precompile_configs(weights_data_type) {
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
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
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
