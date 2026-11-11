use std::sync::OnceLock;

use super::RadixTopKSmallMetalKernel;
use crate::backends::{
    common::{
        BufferMut, BufferRef,
        kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
    },
    metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError},
};

pub struct MetalRadixTopKSmall {
    kernels: [OnceLock<RadixTopKSmallMetalKernel>; 2],
}

impl RadixTopKSmall for MetalRadixTopKSmall {
    type Backend = Metal;

    fn new(context: &MetalContext) -> Result<Self, MetalError> {
        let kernels = [OnceLock::new(), OnceLock::new()];
        kernels[1].set(RadixTopKSmallMetalKernel::new(context, true)?).ok();
        Ok(Self {
            kernels,
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = Metal>,
        output_ids: impl BufferMut<Backend = Metal>,
        output_scores: impl BufferMut<Backend = Metal>,
        rows: u32,
        columns: u32,
        k: u32,
        sorted: bool,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        assert!(rows > 0 && columns > 0 && k > 0 && k <= MAX_K && k <= columns);
        let context = command_buffer.context.clone();
        let kernel = self.kernels[usize::from(sorted)].get_or_init(|| {
            RadixTopKSmallMetalKernel::new(&context, sorted).expect("Failed to create radix top-k kernel")
        });
        kernel.encode(input, output_ids, output_scores, rows, columns, k, command_buffer);
        Ok(())
    }
}
