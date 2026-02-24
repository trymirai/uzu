use std::collections::HashMap;

use super::{
    super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor, specialization::Specialization,
};
use crate::{
    DataType,
    backends::common::{Backend, Kernels, kernel::MatmulGemmMppKernel},
};

pub struct GemmMppKernel<B: Backend> {
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
    pipelines: HashMap<Specialization, <B::Kernels as Kernels>::MatmulGemmMppKernel>,
}

impl<B: Backend> GemmMppKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(
        a_dtype: DataType,
        b_dtype: DataType,
        output_dtype: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            a_dtype,
            b_dtype,
            output_dtype,
            pipelines: HashMap::new(),
        })
    }

    pub fn precompile(
        &mut self,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        for &config in Specialization::precompile_configs(self.output_dtype) {
            self.get_or_create_kernel(context, config)?;
        }
        Ok(())
    }

    fn get_or_create_kernel(
        &mut self,
        context: &B::Context,
        config: Specialization,
    ) -> Result<&<B::Kernels as Kernels>::MatmulGemmMppKernel, B::Error> {
        if !self.pipelines.contains_key(&config) {
            let pipeline = <B::Kernels as Kernels>::MatmulGemmMppKernel::new(
                context,
                self.a_dtype,
                self.b_dtype,
                self.output_dtype,
                config.block_rows as u32,
                config.block_cols as u32,
                config.block_depth as u32,
                config.warps_per_row as u32,
                config.warps_per_col as u32,
                config.align_m,
                config.align_n,
                config.align_k,
            )?;
            self.pipelines.insert(config, pipeline);
        }
        Ok(self.pipelines.get(&config).unwrap())
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        let config = dispatch_descriptor.specialization;

        let group_count_x = u32::try_from(dispatch_descriptor.threadgroups.x).map_err(|_| {
            B::Error::from(format!("GemmMpp group count x overflows u32: {}", dispatch_descriptor.threadgroups.x))
        })?;
        let group_count_y = u32::try_from(dispatch_descriptor.threadgroups.y).map_err(|_| {
            B::Error::from(format!("GemmMpp group count y overflows u32: {}", dispatch_descriptor.threadgroups.y))
        })?;
        let group_count_z = u32::try_from(dispatch_descriptor.threadgroups.z).map_err(|_| {
            B::Error::from(format!("GemmMpp group count z overflows u32: {}", dispatch_descriptor.threadgroups.z))
        })?;

        let pipeline = self.get_or_create_kernel(context, config)?;
        pipeline.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            arguments.d,
            std::slice::from_ref(&dispatch_descriptor.params),
            group_count_x,
            group_count_y,
            group_count_z,
            encoder,
        );

        Ok(())
    }
}
