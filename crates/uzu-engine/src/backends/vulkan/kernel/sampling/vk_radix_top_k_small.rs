use crate::backends::{
    common::{BufferMut, BufferRef, CommandBufferEncoding, kernel::radix_top_k_small::RadixTopKSmall},
    vulkan::{
        Error, RadixTopKSmallCollectVulkanKernel, RadixTopKSmallPassVulkanKernel, RadixTopKSmallSortVulkanKernel,
        VkBuffer, VkCommandBufferEncoding, VkContext, Vulkan,
        gpu_types::radix_top_k::{RADIX_TOP_K_BUCKETS, RADIX_TOP_K_PASSES},
    },
};

/// The columns of a row each workgroup scans, so a row spreads over enough workgroups to use the whole GPU.
const COLUMNS_PER_PARTITION: u32 = 8192;

pub struct VkRadixTopKSmall {
    pass: RadixTopKSmallPassVulkanKernel,
    collect: RadixTopKSmallCollectVulkanKernel,
    sort: RadixTopKSmallSortVulkanKernel,
}

impl RadixTopKSmall for VkRadixTopKSmall {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        columns: u32,
    ) -> Result<Self, Error> {
        let partitions = columns.div_ceil(COLUMNS_PER_PARTITION);
        let context = context.arc();
        Ok(Self {
            pass: RadixTopKSmallPassVulkanKernel::new(&context, columns, partitions)?,
            collect: RadixTopKSmallCollectVulkanKernel::new(&context, columns, partitions)?,
            sort: RadixTopKSmallSortVulkanKernel::new(&context)?,
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
        let mut histograms = command_buffer.allocate_scratch(
            rows as usize * RADIX_TOP_K_PASSES as usize * RADIX_TOP_K_BUCKETS as usize * size_of::<u32>(),
        )?;
        let mut counts = command_buffer.allocate_scratch(rows as usize * size_of::<u32>())?;
        let keys = command_buffer.allocate_scratch(rows as usize * k as usize * 2 * size_of::<u32>())?;
        CommandBufferEncoding::encode_fill(command_buffer, &mut histograms, 0);
        CommandBufferEncoding::encode_fill(command_buffer, &mut counts, 0);
        let (output_ids, output_ids_range) = output_ids.parts();
        let (output_scores, output_scores_range) = output_scores.parts();
        // SAFETY: the common contract makes callers pass [rows, columns] input of the kernels' type and u32/f32
        // [rows, k] outputs that do not alias; the scratch above holds each row's histogram of every pass, its count and
        // its k keys, which the kernels index alone.
        unsafe {
            for pass in 0..RADIX_TOP_K_PASSES {
                self.pass.encode(
                    VkBuffer::resolve_ref(input),
                    VkBuffer::resolve_ref(&histograms),
                    rows,
                    k,
                    pass,
                    command_buffer,
                );
            }
            self.collect.encode(
                VkBuffer::resolve_ref(input),
                VkBuffer::resolve_ref(&histograms),
                VkBuffer::resolve_ref(&counts),
                VkBuffer::resolve_ref(&keys),
                rows,
                k,
                command_buffer,
            );
            self.sort.encode(
                VkBuffer::resolve_ref(&keys),
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
