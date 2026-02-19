use std::collections::HashMap;

use metal::MTLComputeCommandEncoder;
use objc2::runtime::ProtocolObject;

use super::{
    dispatch_descriptor::{DispatchDescriptor, OutputSource},
    pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::{
        common::kernel::MatmulGemvKernel,
        metal::{
            MetalContext, MetalError,
            kernel::{dsl::MatmulGemvMetalKernel, matmul::common::MatmulArguments},
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, MatmulGemvMetalKernel>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported data type for GEMV: {:?}", data_type)));
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
                PipelineConfiguration { threadgroup_rows: 4, threadgroup_cols: 1, threads_per_simdgroup_row: 1, threads_per_simdgroup_col: 32, elements_per_thread_row: 4, elements_per_thread_col: 4, apply_output_scale_and_accumulate: false },
                PipelineConfiguration { threadgroup_rows: 4, threadgroup_cols: 1, threads_per_simdgroup_row: 1, threads_per_simdgroup_col: 32, elements_per_thread_row: 4, elements_per_thread_col: 4, apply_output_scale_and_accumulate: true },
                PipelineConfiguration { threadgroup_rows: 8, threadgroup_cols: 1, threads_per_simdgroup_row: 1, threads_per_simdgroup_col: 32, elements_per_thread_row: 4, elements_per_thread_col: 4, apply_output_scale_and_accumulate: false },
                PipelineConfiguration { threadgroup_rows: 8, threadgroup_cols: 1, threads_per_simdgroup_row: 1, threads_per_simdgroup_col: 32, elements_per_thread_row: 4, elements_per_thread_col: 4, apply_output_scale_and_accumulate: true },
            ],
            DataType::F16 => &[
                PipelineConfiguration { threadgroup_rows: 8, threadgroup_cols: 1, threads_per_simdgroup_row: 1, threads_per_simdgroup_col: 32, elements_per_thread_row: 4, elements_per_thread_col: 4, apply_output_scale_and_accumulate: false },
            ],
            DataType::F32 => &[
                PipelineConfiguration { threadgroup_rows: 8, threadgroup_cols: 1, threads_per_simdgroup_row: 1, threads_per_simdgroup_col: 32, elements_per_thread_row: 4, elements_per_thread_col: 4, apply_output_scale_and_accumulate: false },
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
    ) -> Result<&MatmulGemvMetalKernel, MetalError> {
        if !self.pipelines.contains_key(&config) {
            let kernel = MatmulGemvMetalKernel::new(
                context,
                self.data_type,
                config.threadgroup_rows,
                config.threadgroup_cols,
                config.threads_per_simdgroup_row,
                config.threads_per_simdgroup_col,
                config.elements_per_thread_row,
                config.elements_per_thread_col,
                config.apply_output_scale_and_accumulate,
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
        let pipeline = self.get_or_create_pipeline(context, config)?;

        let (matrix, matrix_offset) = if descriptor.matrix_is_rhs {
            (arguments.b, 0usize)
        } else {
            (arguments.a, arguments.a_offset as usize)
        };
        let (input_vector, input_vector_offset) = if descriptor.matrix_is_rhs {
            (arguments.a, arguments.a_offset as usize)
        } else {
            (arguments.b, 0usize)
        };

        let output_source = if config.apply_output_scale_and_accumulate {
            match descriptor.output_source {
                OutputSource::None => {
                    return Err(MetalError::Generic(
                        "GEMV descriptor mismatch: apply_output_scale_and_accumulate=true but output_source=None"
                            .to_owned(),
                    ));
                },
                OutputSource::Bias => {
                    Some(arguments.bias.ok_or_else(|| {
                        MetalError::Generic("GEMV descriptor requires bias buffer".to_owned())
                    })?)
                },
                OutputSource::C => Some(arguments.c.ok_or_else(|| {
                    MetalError::Generic("GEMV descriptor requires C buffer".to_owned())
                })?),
            }
        } else {
            None
        };

        let vector_batch_stride = [i32::try_from(descriptor.vector_batch_stride[0]).map_err(|_| {
            MetalError::Generic(format!(
                "GEMV path requires i32 vector_batch_stride but got {}",
                descriptor.vector_batch_stride[0]
            ))
        })?];
        let matrix_batch_stride = [i32::try_from(descriptor.matrix_batch_stride[0]).map_err(|_| {
            MetalError::Generic(format!(
                "GEMV path requires i32 matrix_batch_stride but got {}",
                descriptor.matrix_batch_stride[0]
            ))
        })?];
        let output_source_batch_stride = [i32::try_from(descriptor.bias_batch_stride[0]).map_err(|_| {
            MetalError::Generic(format!(
                "GEMV path requires i32 output_source_batch_stride but got {}",
                descriptor.bias_batch_stride[0]
            ))
        })?];

        pipeline.encode(
            (matrix, matrix_offset),
            (input_vector, input_vector_offset),
            output_source.map(|buffer| (buffer, 0usize)),
            arguments.d,
            descriptor.input_dimension,
            descriptor.output_dimension,
            descriptor.matrix_leading_dim,
            descriptor.alpha,
            descriptor.beta,
            &descriptor.batch_shape,
            &vector_batch_stride,
            &matrix_batch_stride,
            &output_source_batch_stride,
            descriptor.bias_stride,
            descriptor.batch_rows,
            config.output_rows_per_threadgroup() as i32,
            encoder,
        );

        Ok(descriptor.bias_is_fused())
    }
}
