use std::{collections::HashMap, ops::DerefMut};

use super::{
    super::matmul_arguments::MatmulArguments, dispatch_descriptor::GemmDispatchDescriptor,
    specialization::Specialization,
};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{MatmulGemmKernel, matmul::MatmulError},
    },
};

pub struct GemmKernel<B: Backend> {
    data_type: DataType,
    pipelines: HashMap<Specialization, <B::Kernels as Kernels>::MatmulGemmKernel>,
}

impl<B: Backend> GemmKernel<B> {
    pub fn new(data_type: DataType) -> Result<Self, MatmulError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
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
    ) -> Result<&<B::Kernels as Kernels>::MatmulGemmKernel, MatmulError<B>> {
        if !self.pipelines.contains_key(&config) {
            let kernel = <B::Kernels as Kernels>::MatmulGemmKernel::new(
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
            )
            .map_err(MatmulError::BackendError)?;
            self.pipelines.insert(config, kernel);
        }
        Ok(self.pipelines.get(&config).unwrap())
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &mut MatmulArguments<B>,
        dispatch_descriptor: &GemmDispatchDescriptor,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<B>> {
        let config = dispatch_descriptor.specialization;

        let group_count_x = u32::try_from(dispatch_descriptor.threadgroups.x)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.x))?;
        let group_count_y = u32::try_from(dispatch_descriptor.threadgroups.y)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.y))?;
        let group_count_z = u32::try_from(dispatch_descriptor.threadgroups.z)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.z))?;

        let pipeline = self.get_or_create_kernel(context, config)?;
        pipeline.encode(
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
