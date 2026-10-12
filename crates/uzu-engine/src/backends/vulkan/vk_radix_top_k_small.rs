use super::{Error, RadixTopKSmallVulkanKernel, VkBuffer, VkCommandBufferEncoding, VkContext, Vulkan};
use crate::backends::common::{
    BufferMut, BufferRef,
    kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
};

pub struct VkRadixTopKSmall {
    kernel: RadixTopKSmallVulkanKernel,
    columns: u32,
}

impl RadixTopKSmall for VkRadixTopKSmall {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        columns: u32,
    ) -> Result<Self, Error> {
        Ok(Self {
            kernel: RadixTopKSmallVulkanKernel::new(&context.arc(), columns)?,
            columns,
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = Vulkan>,
        output_ids: impl BufferMut<Backend = Vulkan>,
        output_scores: impl BufferMut<Backend = Vulkan>,
        rows: u32,
        k: u32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<(), Error> {
        if rows == 0 || k == 0 || k > MAX_K.min(self.columns) {
            return Err(Error::KernelPrecondition {
                kernel: "RadixTopKSmall",
                condition: "rows > 0 && k > 0 && k <= 512 && k <= columns",
            });
        }
        let (input, input_range) = input.parts();
        let (output_ids, output_ids_range) = output_ids.parts();
        let (output_scores, output_scores_range) = output_scores.parts();
        // SAFETY: the common contract makes callers pass bf16 [rows, columns] input and u32/f32 [rows, k] outputs
        // that do not alias, which covers every element the kernel indexes.
        unsafe {
            self.kernel.encode(
                VkBuffer::resolve(input, input_range),
                VkBuffer::resolve(output_ids, output_ids_range),
                VkBuffer::resolve(output_scores, output_scores_range),
                rows,
                k,
                command_buffer,
            );
        }
        Ok(())
    }
}
