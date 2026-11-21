//! Vulkan has no 64-bit integers in kernel arguments, so these wrappers pass the PRNG seed as two words.

use crate::backends::{
    common::{
        BufferMut, BufferRef,
        kernel::{LinearizeChainKernel, LinearizeTreeKernel},
    },
    vulkan::{
        Error, LinearizeChainVulkanKernel, LinearizeTreeVulkanKernel, VkBuffer, VkCommandBufferEncoding, VkContext,
        Vulkan,
    },
};

fn seed_words(seed: u64) -> (u32, u32) {
    (seed as u32, (seed >> 32) as u32)
}

pub struct VkLinearizeChain(LinearizeChainVulkanKernel);

impl LinearizeChainKernel for VkLinearizeChain {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        with_seeds: bool,
    ) -> Result<Self, Error> {
        LinearizeChainVulkanKernel::new(context, with_seeds).map(Self)
    }

    fn encode(
        &self,
        sampled: impl BufferRef<Backend = Vulkan>,
        target_output_token: impl BufferRef<Backend = Vulkan>,
        nodes: impl BufferMut<Backend = Vulkan>,
        token_ids: impl BufferMut<Backend = Vulkan>,
        seeds: Option<impl BufferMut<Backend = Vulkan>>,
        chain_length: u32,
        budget: u32,
        prng_seed: u64,
        root_position: u32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        let (nodes, nodes_range) = nodes.parts();
        let (token_ids, token_ids_range) = token_ids.parts();
        let seeds = seeds.map(BufferMut::parts);
        let (prng_seed_low, prng_seed_high) = seed_words(prng_seed);
        self.0.encode(
            VkBuffer::resolve_ref(sampled),
            VkBuffer::resolve_ref(target_output_token),
            VkBuffer::resolve(nodes, nodes_range),
            VkBuffer::resolve(token_ids, token_ids_range),
            seeds.map(|(seeds, range)| VkBuffer::resolve(seeds, range)),
            chain_length,
            budget,
            prng_seed_low,
            prng_seed_high,
            root_position,
            command_buffer,
        );
    }
}

pub struct VkLinearizeTree(LinearizeTreeVulkanKernel);

impl LinearizeTreeKernel for VkLinearizeTree {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        with_seeds: bool,
    ) -> Result<Self, Error> {
        LinearizeTreeVulkanKernel::new(context, with_seeds).map(Self)
    }

    fn encode(
        &self,
        packed_tree: impl BufferRef<Backend = Vulkan>,
        frontier: impl BufferRef<Backend = Vulkan>,
        nodes: impl BufferMut<Backend = Vulkan>,
        token_ids: impl BufferMut<Backend = Vulkan>,
        seeds: Option<impl BufferMut<Backend = Vulkan>>,
        // The Vulkan kernel keeps its slot-to-flat map in shared memory instead.
        _stack: impl BufferMut<Backend = Vulkan>,
        tree_slot_count: u32,
        frontier_capacity: u32,
        budget: u32,
        prng_seed: u64,
        root_position: u32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        let (nodes, nodes_range) = nodes.parts();
        let (token_ids, token_ids_range) = token_ids.parts();
        let seeds = seeds.map(BufferMut::parts);
        let (prng_seed_low, prng_seed_high) = seed_words(prng_seed);
        self.0.encode(
            VkBuffer::resolve_ref(packed_tree),
            VkBuffer::resolve_ref(frontier),
            VkBuffer::resolve(nodes, nodes_range),
            VkBuffer::resolve(token_ids, token_ids_range),
            seeds.map(|(seeds, range)| VkBuffer::resolve(seeds, range)),
            tree_slot_count,
            frontier_capacity,
            budget,
            prng_seed_low,
            prng_seed_high,
            root_position,
            command_buffer,
        );
    }
}
