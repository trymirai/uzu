use std::collections::HashMap;

use super::{
    super::matmul_arguments::MatmulArguments,
    dispatch_descriptor::{DispatchDescriptor, OutputSource},
    specialization::Specialization,
};
use crate::{
    DataType,
    backends::common::{Backend, Kernels, kernel::MatmulGemvKernel},
};

pub struct GemvKernel<B: Backend> {
    data_type: DataType,
    pipelines: HashMap<Specialization, <B::Kernels as Kernels>::MatmulGemvKernel>,
}

impl<B: Backend> GemvKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(data_type: DataType) -> Result<Self, B::Error> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(B::Error::from(format!("Unsupported data type for GEMV: {data_type:?}")));
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
    ) -> Result<&<B::Kernels as Kernels>::MatmulGemvKernel, B::Error> {
        if !self.pipelines.contains_key(&config) {
            let kernel = <B::Kernels as Kernels>::MatmulGemvKernel::new(
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

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        let config = dispatch_descriptor.specialization;
        let pipeline = self.get_or_create_kernel(context, config)?;

        let (matrix, matrix_offset) = if dispatch_descriptor.matrix_is_rhs {
            (arguments.b, 0usize)
        } else {
            (arguments.a, arguments.a_offset as usize)
        };
        let (input_vector, input_vector_offset) = if dispatch_descriptor.matrix_is_rhs {
            (arguments.a, arguments.a_offset as usize)
        } else {
            (arguments.b, 0usize)
        };

        let output_source = if config.apply_output_scale_and_accumulate {
            match dispatch_descriptor.output_source {
                OutputSource::None => {
                    return Err(B::Error::from(
                        "GEMV descriptor mismatch: apply_output_scale_and_accumulate=true but output_source=None"
                            .to_owned(),
                    ));
                },
                OutputSource::Bias => Some(
                    arguments.bias.ok_or_else(|| B::Error::from("GEMV descriptor requires bias buffer".to_owned()))?,
                ),
                OutputSource::C => {
                    Some(arguments.c.ok_or_else(|| B::Error::from("GEMV descriptor requires C buffer".to_owned()))?)
                },
            }
        } else {
            None
        };

        let vector_batch_stride = [i32::try_from(dispatch_descriptor.vector_batch_stride[0]).map_err(|_| {
            B::Error::from(format!(
                "GEMV path requires i32 vector_batch_stride but got {}",
                dispatch_descriptor.vector_batch_stride[0]
            ))
        })?];
        let matrix_batch_stride = [i32::try_from(dispatch_descriptor.matrix_batch_stride[0]).map_err(|_| {
            B::Error::from(format!(
                "GEMV path requires i32 matrix_batch_stride but got {}",
                dispatch_descriptor.matrix_batch_stride[0]
            ))
        })?];
        let output_source_batch_stride = [i32::try_from(dispatch_descriptor.bias_batch_stride[0]).map_err(|_| {
            B::Error::from(format!(
                "GEMV path requires i32 output_source_batch_stride but got {}",
                dispatch_descriptor.bias_batch_stride[0]
            ))
        })?];

        pipeline.encode(
            (matrix, matrix_offset),
            (input_vector, input_vector_offset),
            output_source.map(|buffer| (buffer, 0usize)),
            arguments.d,
            dispatch_descriptor.input_dimension,
            dispatch_descriptor.output_dimension,
            dispatch_descriptor.matrix_leading_dim,
            dispatch_descriptor.alpha,
            dispatch_descriptor.beta,
            &dispatch_descriptor.batch_shape,
            &vector_batch_stride,
            &matrix_batch_stride,
            &output_source_batch_stride,
            dispatch_descriptor.bias_stride,
            dispatch_descriptor.batch_rows,
            config.output_rows_per_threadgroup() as i32,
            encoder,
        );

        Ok(())
    }
}
