use crate::{
    backends::{
        common::{
            BufferMut, BufferRef, CommandBufferEncoding,
            kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
        },
        vulkan::{
            Error, RadixTopKSmallCollectVulkanKernel, RadixTopKSmallPassVulkanKernel, RadixTopKSmallSortVulkanKernel,
            VkBuffer, VkCommandBufferEncoding, VkContext, Vulkan,
            gpu_types::radix_top_k::{RADIX_TOP_K_BUCKETS, RADIX_TOP_K_MAX_K, RADIX_TOP_K_PASSES},
        },
    },
    data_type::DataType,
};

const _: () = assert!(RADIX_TOP_K_MAX_K == MAX_K);

const COLUMNS_PER_PARTITION: u32 = 8192;

pub struct VkRadixTopKSmall {
    pass: RadixTopKSmallPassVulkanKernel,
    collect: RadixTopKSmallCollectVulkanKernel,
    sort: [RadixTopKSmallSortVulkanKernel; 2],
}

impl RadixTopKSmall for VkRadixTopKSmall {
    type Backend = Vulkan;

    fn new(context: &VkContext) -> Result<Self, Error> {
        Ok(Self {
            pass: RadixTopKSmallPassVulkanKernel::new(context)?,
            collect: RadixTopKSmallCollectVulkanKernel::new(context)?,
            sort: [
                RadixTopKSmallSortVulkanKernel::new(context, false)?,
                RadixTopKSmallSortVulkanKernel::new(context, true)?,
            ],
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = Vulkan>,
        output_ids: impl BufferMut<Backend = Vulkan>,
        output_scores: impl BufferMut<Backend = Vulkan>,
        rows: u32,
        columns: u32,
        k: u32,
        sorted: bool,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<(), Error> {
        assert!(rows > 0 && columns > 0 && k > 0 && k <= MAX_K && k <= columns);
        let partitions = columns.div_ceil(COLUMNS_PER_PARTITION);
        let mut histograms = command_buffer
            .allocate_scratch_for_shape(&[rows, RADIX_TOP_K_PASSES, RADIX_TOP_K_BUCKETS], DataType::U32)?;
        let mut counts = command_buffer.allocate_scratch_for_shape(&[rows], DataType::U32)?;
        let keys = command_buffer.allocate_scratch_for_shape(&[rows, k, 2], DataType::U32)?;
        command_buffer.encode_fill(&mut histograms, 0);
        command_buffer.encode_fill(&mut counts, 0);
        let (output_ids, output_ids_range) = output_ids.parts();
        let (output_scores, output_scores_range) = output_scores.parts();
        for pass in 0..RADIX_TOP_K_PASSES {
            self.pass.encode(
                VkBuffer::resolve_ref(input),
                VkBuffer::resolve_ref(&histograms),
                rows,
                k,
                pass,
                columns,
                partitions,
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
            columns,
            partitions,
            command_buffer,
        );
        self.sort[usize::from(sorted)].encode(
            VkBuffer::resolve_ref(&keys),
            VkBuffer::resolve(output_ids, output_ids_range),
            VkBuffer::resolve(output_scores, output_scores_range),
            rows,
            k,
            command_buffer,
        );
        Ok(())
    }
}
