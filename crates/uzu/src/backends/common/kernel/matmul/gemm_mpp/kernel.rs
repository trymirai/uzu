use std::{collections::hash_map::Entry, collections::HashMap, ops::DerefMut};

use super::{
    super::{MatmulError, matmul_arguments::MatmulArguments},
    dispatch_descriptor::DispatchDescriptor,
    specialization::Specialization,
};
use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer, Kernels, kernel::MatmulGemmMppKernel},
};

pub struct GemmMppKernel<B: Backend> {
    data_type: DataType,
    kernels: HashMap<Specialization, <B::Kernels as Kernels>::MatmulGemmMppKernel>,
}

impl<B: Backend> GemmMppKernel<B> {
    pub fn new(data_type: DataType) -> Result<Self, B::Error> {
        Ok(Self {
            data_type,
            kernels: HashMap::new(),
        })
    }

    pub fn precompile(
        &mut self,
        context: &B::Context,
    ) -> Result<(), MatmulError<B>> {
        for &config in Specialization::precompile_configs(self.data_type) {
            self.get_or_create_kernel(context, config)?;
        }
        Ok(())
    }

    fn get_or_create_kernel(
        &mut self,
        context: &B::Context,
        config: Specialization,
    ) -> Result<&<B::Kernels as Kernels>::MatmulGemmMppKernel, MatmulError<B>> {
        match self.kernels.entry(config) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = <B::Kernels as Kernels>::MatmulGemmMppKernel::new(
                    context,
                    self.data_type,
                    config.block_rows as u32,
                    config.block_cols as u32,
                    config.block_depth as u32,
                    config.warps_per_row as u32,
                    config.warps_per_col as u32,
                    config.align_m,
                    config.align_n,
                    config.align_k,
                    config.use_native_fragment_layout,
                    config.subtile_rows,
                    config.subtile_cols,
                    config.matmul_k_step,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            }
        }
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &mut MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<B>> {
        let config = dispatch_descriptor.specialization;

        let group_count_x = u32::try_from(dispatch_descriptor.threadgroups.x)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.x))?;
        let group_count_y = u32::try_from(dispatch_descriptor.threadgroups.y)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.y))?;
        let group_count_z = u32::try_from(dispatch_descriptor.threadgroups.z)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.z))?;

        let kernel = self.get_or_create_kernel(context, config)?;
        kernel.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            arguments.d.deref_mut(),
            std::slice::from_ref(&dispatch_descriptor.params),
            group_count_x,
            group_count_y,
            group_count_z,
            command_buffer,
        );

        Ok(())
    }
}
