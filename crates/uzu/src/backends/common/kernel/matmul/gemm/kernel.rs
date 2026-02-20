use std::collections::HashMap;

use super::{
    super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor, specialization::Specialization,
};
use crate::{
    DataType,
    backends::common::{Backend, Kernels, kernel::MatmulGemmKernel},
};

pub struct GemmKernel<B: Backend> {
    data_type: DataType,
    pipelines: HashMap<Specialization, <B::Kernels as Kernels>::MatmulGemmKernel>,
}

impl<B: Backend> GemmKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(data_type: DataType) -> Result<Self, B::Error> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(B::Error::from(format!("Unsupported dtype for GEMM: {data_type:?}")));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    pub fn precompile(
        &mut self,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        for &config in Specialization::precompile_configs(self.data_type) {
            self.get_or_create_kernel(context, config)?;
        }
        Ok(())
    }

    fn get_or_create_kernel(
        &mut self,
        context: &B::Context,
        config: Specialization,
    ) -> Result<&<B::Kernels as Kernels>::MatmulGemmKernel, B::Error> {
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
            )?;
            self.pipelines.insert(config, kernel);
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
            B::Error::from(format!("GEMM group count x overflows u32: {}", dispatch_descriptor.threadgroups.x))
        })?;
        let group_count_y = u32::try_from(dispatch_descriptor.threadgroups.y).map_err(|_| {
            B::Error::from(format!("GEMM group count y overflows u32: {}", dispatch_descriptor.threadgroups.y))
        })?;
        let group_count_z = u32::try_from(dispatch_descriptor.threadgroups.z).map_err(|_| {
            B::Error::from(format!("GEMM group count z overflows u32: {}", dispatch_descriptor.threadgroups.z))
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
