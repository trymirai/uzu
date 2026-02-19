use std::collections::HashMap;

use metal::MTLComputeCommandEncoder;
use objc2::runtime::ProtocolObject;

use super::{DispatchDescriptor, pipeline_configuration::PipelineConfiguration};
use crate::{
    DataType,
    backends::{
        common::kernel::MatmulGemmKernel,
        metal::{
            context::MetalContext,
            error::MetalError,
            kernel::{dsl::MatmulGemmMetalKernel, matmul::common::MatmulArguments},
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, MatmulGemmMetalKernel>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported dtype for GEMM: {data_type:?}")));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    pub fn precompile(
        &mut self,
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        let configs: &[PipelineConfiguration] = match self.data_type {
            DataType::BF16 => &[
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 32,
                    block_depth: 32,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 32,
                    block_depth: 32,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: false,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 1,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
            ],
            DataType::F16 => &[
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 2,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
            ],
            DataType::F32 => &[
                PipelineConfiguration {
                    block_rows: 32,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 1,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
                PipelineConfiguration {
                    block_rows: 32,
                    block_cols: 64,
                    block_depth: 16,
                    warps_per_row: 1,
                    warps_per_col: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
            ],
            _ => &[],
        };

        for &config in configs {
            let _ = self.get_or_create_pipeline(context, config)?;
        }

        Ok(())
    }

    fn get_or_create_pipeline(
        &mut self,
        context: &MetalContext,
        config: PipelineConfiguration,
    ) -> Result<&MatmulGemmMetalKernel, MetalError> {
        if !self.pipelines.contains_key(&config) {
            let kernel = MatmulGemmMetalKernel::new(
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

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MetalError> {
        let config = descriptor.pipeline_configuration;

        let group_count_x = u32::try_from(descriptor.threadgroups.width).map_err(|_| {
            MetalError::Generic(format!("GEMM group count x overflows u32: {}", descriptor.threadgroups.width))
        })?;
        let group_count_y = u32::try_from(descriptor.threadgroups.height).map_err(|_| {
            MetalError::Generic(format!("GEMM group count y overflows u32: {}", descriptor.threadgroups.height))
        })?;
        let group_count_z = u32::try_from(descriptor.threadgroups.depth).map_err(|_| {
            MetalError::Generic(format!("GEMM group count z overflows u32: {}", descriptor.threadgroups.depth))
        })?;

        let pipeline = self.get_or_create_pipeline(context, config)?;
        pipeline.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            arguments.d,
            std::slice::from_ref(&descriptor.params),
            group_count_x,
            group_count_y,
            group_count_z,
            encoder,
        );

        Ok(false)
    }
}
