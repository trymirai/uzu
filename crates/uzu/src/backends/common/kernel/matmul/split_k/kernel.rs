use std::ops::{Deref, DerefMut};

use super::{super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Context, Kernels,
        kernel::{MatmulSplitKAccumBfloat16Kernel, MatmulSplitKPartialBfloat16Kernel, matmul::MatmulError},
    },
};

pub struct SplitKKernel<B: Backend> {
    data_type: DataType,
    partial_bfloat16: Option<<B::Kernels as Kernels>::MatmulSplitKPartialBfloat16Kernel>,
    accum_bfloat16: Option<<B::Kernels as Kernels>::MatmulSplitKAccumBfloat16Kernel>,
    accumulator_buffer: Option<B::Buffer>,
    accumulator_buffer_bytes: usize,
}

impl<B: Backend> SplitKKernel<B> {
    pub fn new(data_type: DataType) -> Result<Self, MatmulError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }
        Ok(Self {
            data_type,
            partial_bfloat16: None,
            accum_bfloat16: None,
            accumulator_buffer: None,
            accumulator_buffer_bytes: 0,
        })
    }

    pub fn precompile(
        &mut self,
        context: &B::Context,
    ) -> Result<(), MatmulError<B>> {
        if !matches!(self.data_type, DataType::BF16) {
            return Ok(());
        }
        self.ensure_kernels(context)
    }

    fn ensure_kernels(
        &mut self,
        context: &B::Context,
    ) -> Result<(), MatmulError<B>> {
        if !matches!(self.data_type, DataType::BF16) {
            return Err(MatmulError::UnsupportedDataType(self.data_type));
        }
        if self.partial_bfloat16.is_none() {
            self.partial_bfloat16 = Some(
                <B::Kernels as Kernels>::MatmulSplitKPartialBfloat16Kernel::new(context)
                    .map_err(MatmulError::BackendError)?,
            );
        }
        if self.accum_bfloat16.is_none() {
            self.accum_bfloat16 = Some(
                <B::Kernels as Kernels>::MatmulSplitKAccumBfloat16Kernel::new(context)
                    .map_err(MatmulError::BackendError)?,
            );
        }
        Ok(())
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &mut MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<B>> {
        self.ensure_kernels(context)?;
        self.ensure_accumulator_buffer(context, dispatch_descriptor.accumulator_bytes)?;
        let mut accumulator_buffer = self.accumulator_buffer.as_mut().expect("Accumulator buffer must be initialized");

        let partial_group_count_x = u32::try_from(dispatch_descriptor.partial_threadgroups.x)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.partial_threadgroups.x))?;
        let partial_group_count_y = u32::try_from(dispatch_descriptor.partial_threadgroups.y)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.partial_threadgroups.y))?;
        let partial_group_count_z = u32::try_from(dispatch_descriptor.partial_threadgroups.z)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.partial_threadgroups.z))?;
        let partial = self.partial_bfloat16.as_ref().unwrap();
        partial.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            accumulator_buffer.deref_mut(),
            std::slice::from_ref(&dispatch_descriptor.params),
            partial_group_count_x,
            partial_group_count_y,
            partial_group_count_z,
            command_buffer,
        );

        let accum_total_threads_x = u32::try_from(dispatch_descriptor.accum_total_threads.x)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.accum_total_threads.x))?;
        let accum_total_threads_y = u32::try_from(dispatch_descriptor.accum_total_threads.y)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.accum_total_threads.y))?;
        let accum = self.accum_bfloat16.as_ref().unwrap();
        accum.encode(
            accumulator_buffer.deref(),
            arguments.d.deref_mut(),
            dispatch_descriptor.partition_count,
            dispatch_descriptor.output_elements_per_partition,
            arguments.ldd,
            accum_total_threads_x,
            accum_total_threads_y,
            command_buffer,
        );

        Ok(())
    }

    fn ensure_accumulator_buffer(
        &mut self,
        context: &B::Context,
        required_bytes: usize,
    ) -> Result<(), MatmulError<B>> {
        if required_bytes <= self.accumulator_buffer_bytes && self.accumulator_buffer.is_some() {
            return Ok(());
        }
        self.accumulator_buffer = Some(context.create_buffer(required_bytes).map_err(MatmulError::BackendError)?);
        self.accumulator_buffer_bytes = required_bytes;
        Ok(())
    }
}
