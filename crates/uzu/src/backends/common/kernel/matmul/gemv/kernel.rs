use std::{collections::hash_map::Entry, collections::HashMap, ops::DerefMut};

use super::{
    super::matmul_arguments::MatmulArguments,
    dispatch_descriptor::{DispatchDescriptor, OutputSource},
    specialization::Specialization,
};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{MatmulGemvKernel, matmul::MatmulError},
    },
};

pub struct GemvKernel<B: Backend> {
    data_type: DataType,
    kernels: HashMap<Specialization, <B::Kernels as Kernels>::MatmulGemvKernel>,
}

impl<B: Backend> GemvKernel<B> {
    pub fn new(data_type: DataType) -> Result<Self, MatmulError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }
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
    ) -> Result<&<B::Kernels as Kernels>::MatmulGemvKernel, MatmulError<B>> {
        match self.kernels.entry(config) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
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
        let kernel = self.get_or_create_kernel(context, config)?;

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
                    return Err(MatmulError::GemvOutputSourceMismatch);
                },
                OutputSource::Bias => Some(arguments.bias.ok_or(MatmulError::<B>::GemvMissingBias)?),
            }
        } else {
            None
        };

        let vector_batch_stride = [i32::try_from(dispatch_descriptor.vector_batch_stride[0])
            .map_err(|_| MatmulError::<B>::GemvStrideOverflow(dispatch_descriptor.vector_batch_stride[0]))?];
        let matrix_batch_stride = [i32::try_from(dispatch_descriptor.matrix_batch_stride[0])
            .map_err(|_| MatmulError::<B>::GemvStrideOverflow(dispatch_descriptor.matrix_batch_stride[0]))?];
        let output_source_batch_stride = [i32::try_from(dispatch_descriptor.bias_batch_stride[0])
            .map_err(|_| MatmulError::<B>::GemvStrideOverflow(dispatch_descriptor.bias_batch_stride[0]))?];

        kernel.encode(
            (matrix, matrix_offset),
            (input_vector, input_vector_offset),
            output_source.map(|buffer| (buffer, 0usize)),
            arguments.d.deref_mut(),
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
            command_buffer,
        );

        Ok(())
    }
}
