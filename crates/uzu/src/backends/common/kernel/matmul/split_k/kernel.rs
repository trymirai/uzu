use super::{super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor};
use crate::{
    DataType,
    backends::common::{
        Backend, Context, Kernels,
        kernel::{MatmulSplitKAccumBfloat16Kernel, MatmulSplitKPartialBfloat16Kernel},
    },
};

pub struct SplitKKernel<B: Backend> {
    data_type: DataType,
    partial_bfloat16: Option<<B::Kernels as Kernels>::MatmulSplitKPartialBfloat16Kernel>,
    accum_bfloat16: Option<<B::Kernels as Kernels>::MatmulSplitKAccumBfloat16Kernel>,
    accumulator_buffer: Option<B::NativeBuffer>,
    accumulator_buffer_bytes: usize,
}

impl<B: Backend> SplitKKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(data_type: DataType) -> Result<Self, B::Error> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(B::Error::from(format!("Unsupported dtype for Split-K: {data_type:?}")));
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
    ) -> Result<(), B::Error> {
        if !matches!(self.data_type, DataType::BF16) {
            return Ok(());
        }
        self.ensure_kernels(context)
    }

    fn ensure_kernels(
        &mut self,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        if !matches!(self.data_type, DataType::BF16) {
            return Err(B::Error::from(format!("Split-K currently supports BF16 only, got {:?}", self.data_type)));
        }
        if self.partial_bfloat16.is_none() {
            self.partial_bfloat16 = Some(<B::Kernels as Kernels>::MatmulSplitKPartialBfloat16Kernel::new(context)?);
        }
        if self.accum_bfloat16.is_none() {
            self.accum_bfloat16 = Some(<B::Kernels as Kernels>::MatmulSplitKAccumBfloat16Kernel::new(context)?);
        }
        Ok(())
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        self.ensure_kernels(context)?;
        self.ensure_accumulator_buffer(context, dispatch_descriptor.accumulator_bytes)?;
        let accumulator_buffer =
            self.accumulator_buffer.as_ref().cloned().expect("Accumulator buffer must be initialized");

        let partial_group_count_x = u32::try_from(dispatch_descriptor.partial_threadgroups.x).map_err(|_| {
            B::Error::from(format!(
                "Split-K partial group count x overflows u32: {}",
                dispatch_descriptor.partial_threadgroups.x
            ))
        })?;
        let partial_group_count_y = u32::try_from(dispatch_descriptor.partial_threadgroups.y).map_err(|_| {
            B::Error::from(format!(
                "Split-K partial group count y overflows u32: {}",
                dispatch_descriptor.partial_threadgroups.y
            ))
        })?;
        let partial_group_count_z = u32::try_from(dispatch_descriptor.partial_threadgroups.z).map_err(|_| {
            B::Error::from(format!(
                "Split-K partial group count z overflows u32: {}",
                dispatch_descriptor.partial_threadgroups.z
            ))
        })?;
        let partial = self.partial_bfloat16.as_ref().unwrap();
        partial.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &accumulator_buffer,
            std::slice::from_ref(&dispatch_descriptor.params),
            partial_group_count_x,
            partial_group_count_y,
            partial_group_count_z,
            encoder,
        );

        let accum_total_threads_x = u32::try_from(dispatch_descriptor.accum_total_threads.x).map_err(|_| {
            B::Error::from(format!(
                "Split-K accum total threads x overflows u32: {}",
                dispatch_descriptor.accum_total_threads.x
            ))
        })?;
        let accum_total_threads_y = u32::try_from(dispatch_descriptor.accum_total_threads.y).map_err(|_| {
            B::Error::from(format!(
                "Split-K accum total threads y overflows u32: {}",
                dispatch_descriptor.accum_total_threads.y
            ))
        })?;
        let accum = self.accum_bfloat16.as_ref().unwrap();
        accum.encode(
            &accumulator_buffer,
            arguments.d,
            dispatch_descriptor.partition_count,
            dispatch_descriptor.output_elements_per_partition,
            arguments.ldd,
            accum_total_threads_x,
            accum_total_threads_y,
            encoder,
        );

        Ok(())
    }

    fn ensure_accumulator_buffer(
        &mut self,
        context: &B::Context,
        required_bytes: usize,
    ) -> Result<(), B::Error> {
        if required_bytes <= self.accumulator_buffer_bytes && self.accumulator_buffer.is_some() {
            return Ok(());
        }
        self.accumulator_buffer = Some(context.create_buffer(required_bytes)?);
        self.accumulator_buffer_bytes = required_bytes;
        Ok(())
    }
}
