use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::DispatchDescriptor;
use crate::{
    DataType,
    backends::{
        common::{
            Context,
            kernel::{MatmulSplitKAccumBfloat16Kernel, MatmulSplitKPartialBfloat16Kernel},
        },
        metal::{
            MetalContext, MetalError,
            kernel::{
                dsl::{MatmulSplitKAccumBfloat16MetalKernel, MatmulSplitKPartialBfloat16MetalKernel},
                matmul::common::MatmulArguments,
            },
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    partial_bfloat16: Option<MatmulSplitKPartialBfloat16MetalKernel>,
    accum_bfloat16: Option<MatmulSplitKAccumBfloat16MetalKernel>,
    accumulator_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    accumulator_buffer_bytes: usize,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported dtype for Split-K: {:?}", data_type)));
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
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        if !matches!(self.data_type, DataType::BF16) {
            return Ok(());
        }
        self.ensure_kernels(context)
    }

    fn ensure_kernels(
        &mut self,
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        if !matches!(self.data_type, DataType::BF16) {
            return Err(MetalError::Generic(format!("Split-K currently supports BF16 only, got {:?}", self.data_type)));
        }
        if self.partial_bfloat16.is_none() {
            self.partial_bfloat16 = Some(MatmulSplitKPartialBfloat16MetalKernel::new(context)?);
        }
        if self.accum_bfloat16.is_none() {
            self.accum_bfloat16 = Some(MatmulSplitKAccumBfloat16MetalKernel::new(context)?);
        }
        Ok(())
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MetalContext,
        encoder: &ProtocolObject<dyn metal::MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MetalError> {
        self.ensure_kernels(context)?;
        self.ensure_accumulator_buffer(context, descriptor.accumulator_bytes);
        let accumulator_buffer =
            self.accumulator_buffer.as_ref().cloned().expect("Accumulator buffer must be initialized");

        let partial_group_count_x = u32::try_from(descriptor.partial_threadgroups.width).map_err(|_| {
            MetalError::Generic(format!(
                "Split-K partial group count x overflows u32: {}",
                descriptor.partial_threadgroups.width
            ))
        })?;
        let partial_group_count_y = u32::try_from(descriptor.partial_threadgroups.height).map_err(|_| {
            MetalError::Generic(format!(
                "Split-K partial group count y overflows u32: {}",
                descriptor.partial_threadgroups.height
            ))
        })?;
        let partial_group_count_z = u32::try_from(descriptor.partial_threadgroups.depth).map_err(|_| {
            MetalError::Generic(format!(
                "Split-K partial group count z overflows u32: {}",
                descriptor.partial_threadgroups.depth
            ))
        })?;
        let partial = self.partial_bfloat16.as_ref().unwrap();
        partial.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &accumulator_buffer,
            std::slice::from_ref(&descriptor.params),
            partial_group_count_x,
            partial_group_count_y,
            partial_group_count_z,
            encoder,
        );

        let accum_total_threads_x = u32::try_from(descriptor.accum_total_threads.width).map_err(|_| {
            MetalError::Generic(format!(
                "Split-K accum total threads x overflows u32: {}",
                descriptor.accum_total_threads.width
            ))
        })?;
        let accum_total_threads_y = u32::try_from(descriptor.accum_total_threads.height).map_err(|_| {
            MetalError::Generic(format!(
                "Split-K accum total threads y overflows u32: {}",
                descriptor.accum_total_threads.height
            ))
        })?;
        let accum = self.accum_bfloat16.as_ref().unwrap();
        accum.encode(
            &accumulator_buffer,
            arguments.d,
            descriptor.partition_count,
            descriptor.output_elements_per_partition,
            arguments.ldd,
            accum_total_threads_x,
            accum_total_threads_y,
            encoder,
        );

        Ok(false)
    }

    fn ensure_accumulator_buffer(
        &mut self,
        context: &MetalContext,
        required_bytes: usize,
    ) {
        if required_bytes <= self.accumulator_buffer_bytes && self.accumulator_buffer.is_some() {
            return;
        }
        self.accumulator_buffer =
            Some(context.create_buffer(required_bytes).expect("Failed to create accumulator buffer"));
        self.accumulator_buffer_bytes = required_bytes;
    }
}
