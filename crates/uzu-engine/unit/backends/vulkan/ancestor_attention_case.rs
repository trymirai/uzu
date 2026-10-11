use std::{ops::Range, sync::Arc};

use half::bf16;

use super::{
    AttentionPrepareCase, AttentionSinglePassCase, attention_single_pass_case::hashed, kernel_fixture::KernelFixture,
};
use crate::{
    backends::{
        common::{Backend, Context, Kernels, gpu_types::weaver::MetadataIdx, kernel::AncestorAttentionKernel},
        cpu::Cpu,
        vulkan::{AncestorAttentionVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

pub const HEAD_DIM: u32 = 128;

/// One raw AncestorAttention dispatch, a frontier of `rows` nodes, shared by the CPU and Vulkan runners and the gathered
/// oracle: BF16 rows `[rows, 3, heads, 128]`, prefix and node caches of a key plane then a value plane
/// `[length, heads, 128]`, FP32 rotary tables of `max_depth + 1` rows of 128, the depths in the metadata's depth plane
/// (other planes poisoned), each row's listed ancestor slots, `padding` filling its unused ones, and its destination.
#[derive(Clone)]
pub struct AncestorAttentionCase {
    pub num_heads: u32,
    pub prefix_length: u32,
    pub node_capacity: u32,
    pub ancestor_stride: u32,
    pub max_depth: u32,
    pub scale: f32,
    pub depths: Vec<u32>,
    pub ancestors: Vec<Vec<u32>>,
    pub padding: u32,
    pub destinations: Vec<u32>,
    pub prefix_kv: Vec<bf16>,
    pub node_kv: Vec<bf16>,
    pub current_qkv: Vec<bf16>,
    pub cosines: Vec<f32>,
    pub sines: Vec<f32>,
}

impl AncestorAttentionCase {
    /// A Weaver frontier with hashed values in [-2, 2): slot 0 holds quiet and signaling NaNs of either sign and fills
    /// the unused ancestor entries, slots 1 to `node_capacity - rows` the earlier frontiers, which rows list out of order
    /// and repeatedly, up to `stride` each, and the last `rows` slots are the distinct destinations.
    pub fn new(
        num_heads: u32,
        rows: u32,
        prefix_length: u32,
        ancestor_stride: u32,
        seed: u32,
    ) -> Self {
        let model_dim = (num_heads * HEAD_DIM) as usize;
        let node_capacity = rows + ancestor_stride + 2;
        let earlier = node_capacity - rows - 1;
        let max_depth = 8;
        let data = |length: usize, seed: u32| {
            (0..length as u32).map(|index| bf16::from_f32(2.0 * hashed(index, seed))).collect::<Vec<_>>()
        };
        let mut node_kv = data(2 * node_capacity as usize * model_dim, seed + 1);
        let payloads = [0x7fc0, 0xffc1, 0x7f81, 0xff82].map(bf16::from_bits);
        for plane in 0..2 {
            let slot = &mut node_kv[plane * node_capacity as usize * model_dim..][..model_dim];
            slot.iter_mut().enumerate().for_each(|(index, value)| *value = payloads[index % 4]);
        }
        let tables = |seed: u32| (0..(max_depth + 1) * HEAD_DIM).map(|index| hashed(index, seed)).collect::<Vec<_>>();
        Self {
            num_heads,
            prefix_length,
            node_capacity,
            ancestor_stride,
            max_depth,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            depths: (0..rows).map(|row| (row * 3 + seed) % max_depth).collect(),
            ancestors: (0..rows)
                .map(|row| {
                    let count = (row + seed) % (ancestor_stride + 1);
                    (0..count).map(|offset| 1 + (row * 7 + offset * 5 + seed) % earlier).collect()
                })
                .collect(),
            padding: 0,
            destinations: (earlier + 1..node_capacity).collect(),
            prefix_kv: data(2 * prefix_length as usize * model_dim, seed + 2),
            node_kv,
            current_qkv: data(rows as usize * 3 * model_dim, seed + 3),
            cosines: tables(seed + 4),
            sines: tables(seed + 5),
        }
    }

    pub fn rows(&self) -> u32 {
        self.depths.len() as u32
    }

    fn model_dim(&self) -> usize {
        (self.num_heads * HEAD_DIM) as usize
    }

    /// The metadata planes `[MetadataIdx::COUNT, rows]`: depths in the depth plane, the others `u32::MAX`.
    pub fn metadata(&self) -> Vec<u32> {
        let mut metadata = vec![u32::MAX; MetadataIdx::COUNT * self.depths.len()];
        metadata[MetadataIdx::Depth as usize * self.depths.len()..][..self.depths.len()].copy_from_slice(&self.depths);
        metadata
    }

    /// `[ancestor_indices, ancestor_counts, node_indices]`; without node slots the destinations are `u32::MAX`.
    pub fn indices(&self) -> [Vec<u32>; 3] {
        let indices = self
            .ancestors
            .iter()
            .flat_map(|listed| {
                let padding = self.ancestor_stride as usize - listed.len();
                listed.iter().copied().chain(std::iter::repeat_n(self.padding, padding))
            })
            .collect();
        let counts = self.ancestors.iter().map(|listed| listed.len() as u32).collect();
        let destinations = match self.node_capacity {
            0 => vec![u32::MAX; self.depths.len()],
            _ => self.destinations.clone(),
        };
        [indices, counts, destinations]
    }

    /// The rows rotated by the AttentionPrepare oracle against each row's table row `depth + 1`:
    /// `[queries [heads, rows, 128], keys [rows, heads, 128], values [rows, heads, 128]]`.
    pub fn rotated(&self) -> [Vec<bf16>; 3] {
        let gathered = |table: &[f32]| {
            self.depths
                .iter()
                .flat_map(|&depth| table[((depth + 1) * HEAD_DIM) as usize..][..HEAD_DIM as usize].to_vec())
                .collect::<Vec<_>>()
        };
        let prepare = AttentionPrepareCase {
            qkvg: self.current_qkv.clone(),
            cosines: Some(gathered(&self.cosines)),
            sines: Some(gathered(&self.sines)),
            num_q_heads: self.num_heads,
            num_kv_heads: Some(self.num_heads),
            head_dim: HEAD_DIM,
            rope_dim: Some(HEAD_DIM),
            kv_token_offset: Some(0),
            input_row_stride: 3 * self.num_heads * HEAD_DIM,
            batch_dim: self.rows(),
        };
        let length = self.rows() as usize * self.model_dim();
        prepare.oracle().map(|output| output[..length].iter().map(|&(value, _)| value).collect())
    }

    /// The node cache after the dispatch: each row's rotated key and value bits in its destination slot.
    pub fn expected_cache(
        &self,
        [_, keys, values]: &[Vec<bf16>; 3],
    ) -> Vec<bf16> {
        let (model_dim, mut cache) = (self.model_dim(), self.node_kv.clone());
        if self.node_capacity > 0 {
            for (row, &slot) in self.destinations.iter().enumerate() {
                for (plane, source) in [keys, values].into_iter().enumerate() {
                    let start = (plane * self.node_capacity as usize + slot as usize) * model_dim;
                    cache[start..][..model_dim].copy_from_slice(&source[row * model_dim..][..model_dim]);
                }
            }
        }
        cache
    }

    /// The AttentionSinglePass case of `row`: its rotated query over the prefix, its listed ancestors and its own
    /// rotated key and value, unmasked, in the model's cache layout.
    pub fn single_pass(
        &self,
        row: usize,
        [queries, keys, values]: &[Vec<bf16>; 3],
    ) -> AttentionSinglePassCase {
        let (model_dim, rows) = (self.model_dim(), self.depths.len());
        let prefix = self.prefix_length as usize * model_dim;
        let planes = |plane: usize, current: &[bf16]| {
            let node = plane * self.node_capacity as usize * model_dim;
            let ancestors = self.ancestors[row]
                .iter()
                .flat_map(|&slot| &self.node_kv[node + slot as usize * model_dim..][..model_dim]);
            self.prefix_kv[plane * prefix..][..prefix]
                .iter()
                .chain(ancestors)
                .chain(&current[row * model_dim..][..model_dim])
                .map(|value| value.to_f32())
                .collect::<Vec<_>>()
        };
        let head_dim = HEAD_DIM as usize;
        AttentionSinglePassCase {
            head_dim: HEAD_DIM,
            num_heads: self.num_heads,
            gqa_factor: 1,
            prefix_length: self.prefix_length + self.ancestors[row].len() as u32,
            suffix_length: 1,
            k_strides: (HEAD_DIM, self.num_heads * HEAD_DIM),
            v_strides: (HEAD_DIM, self.num_heads * HEAD_DIM),
            ring: None,
            parents: None,
            window: None,
            is_causal: false,
            scale: self.scale,
            queries: (0..self.num_heads as usize)
                .flat_map(|head| &queries[(head * rows + row) * head_dim..][..head_dim])
                .map(|value| value.to_f32())
                .collect(),
            keys: planes(0, keys),
            values: planes(1, values),
            sinks: None,
        }
    }

    /// The CPU kernel through the shared trait over successive frontiers sharing the prefix and the node cache of the
    /// first: each frontier's output, then the node cache. CPU buffers cannot be empty.
    pub fn cpu(frontiers: &[Self]) -> (Vec<Vec<bf16>>, Vec<bf16>) {
        let first = &frontiers[0];
        let context = create_context::<Cpu>();
        let kernel =
            <<Cpu as Backend>::Kernels as Kernels>::AncestorAttentionKernel::new(&context, HEAD_DIM, first.num_heads)
                .expect("CPU AncestorAttention");
        fn buffer<T: crate::array::ArrayElement + Default>(
            context: &Arc<<Cpu as Backend>::Context>,
            values: &[T],
        ) -> <Cpu as Backend>::GlobalBuffer {
            match values.is_empty() {
                true => create_buffer_with_data::<Cpu, T>(context, &[T::default()]),
                false => create_buffer_with_data::<Cpu, T>(context, values),
            }
        }
        let prefix_kv = buffer(&context, &first.prefix_kv);
        let mut node_kv = buffer(&context, &first.node_kv);
        let (cosines, sines) = (buffer(&context, &first.cosines), buffer(&context, &first.sines));
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        // Every buffer outlives the submission, which runs the recorded dispatches.
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for frontier in frontiers {
            let [indices, counts, destinations] = frontier.indices();
            let length = frontier.rows() as usize * frontier.model_dim();
            let mut output = buffer(&context, &vec![bf16::ZERO; length]);
            let words = [frontier.metadata(), indices, counts, destinations].map(|words| buffer(&context, &words));
            inputs.push((buffer(&context, &frontier.current_qkv), words));
            let (current_qkv, [metadata, indices, counts, destinations]) = inputs.last().unwrap();
            kernel.encode(
                &prefix_kv,
                &mut node_kv,
                current_qkv,
                &cosines,
                &sines,
                metadata,
                indices,
                counts,
                destinations,
                &mut output,
                frontier.rows(),
                frontier.prefix_length,
                frontier.ancestor_stride,
                frontier.node_capacity,
                frontier.max_depth,
                frontier.scale,
                &mut command_buffer,
            );
            outputs.push((output, length));
        }
        submit_command_buffer(command_buffer);
        let outputs = outputs
            .into_iter()
            .map(|(output, length)| buffer_to_vec::<Cpu, bf16>(&output)[..length].to_vec())
            .collect();
        let node_kv = buffer_to_vec::<Cpu, bf16>(&node_kv)[..first.node_kv.len()].to_vec();
        (outputs, node_kv)
    }

    pub fn vulkan_kernel(
        &self,
        fixture: &KernelFixture,
    ) -> AncestorAttentionVulkanKernel {
        AncestorAttentionVulkanKernel::new(&fixture.context, HEAD_DIM, self.num_heads)
            .expect("Vulkan AncestorAttention")
    }

    /// Records the frontier's dispatch over `[prefix_kv, node_kv, current_qkv, cosines, sines, node_metadata,
    /// ancestor_indices, ancestor_counts, node_indices, output]`.
    ///
    /// # Safety
    /// The ranges hold every element the frontier indexes, aligned, and the output aliases nothing. The frontier meets
    /// the kernel's caller preconditions: depths below max_depth, at most `ancestor_stride` listed slots per row, all
    /// below node_capacity, as are the destinations, which are distinct and disjoint from every listed slot.
    pub unsafe fn encode(
        &self,
        kernel: &AncestorAttentionVulkanKernel,
        buffers: [(&Arc<VkBuffer>, Range<u64>); 10],
        encoding: &mut VkCommandBufferEncoding,
    ) {
        let [prefix_kv, node_kv, current_qkv, cosines, sines, metadata, indices, counts, destinations, output] =
            buffers;
        // SAFETY: forwarded from the caller.
        unsafe {
            kernel.encode(
                prefix_kv,
                node_kv,
                current_qkv,
                cosines,
                sines,
                metadata,
                indices,
                counts,
                destinations,
                output,
                self.rows(),
                self.prefix_length,
                self.ancestor_stride,
                self.node_capacity,
                self.max_depth,
                self.scale,
                encoding,
            )
        }
    }

    /// Successive frontiers recorded into one command buffer over guarded ranges, sharing the prefix and the node cache
    /// of the first: each frontier's output, then the node cache, after asserting every guard and input unchanged.
    pub fn gpu(
        fixture: &KernelFixture,
        kernel: &AncestorAttentionVulkanKernel,
        frontiers: &[Self],
    ) -> (Vec<Vec<bf16>>, Vec<bf16>) {
        let first = &frontiers[0];
        let (sentinel, word, table) = (bf16::from_bits(0x7bad), 0x5eed_5eed_u32, -7.0f32);
        let prefix_kv = fixture.guarded(&first.prefix_kv, sentinel);
        let node_kv = fixture.guarded(&first.node_kv, sentinel);
        let (cosines, sines) = (fixture.guarded(&first.cosines, table), fixture.guarded(&first.sines, table));
        let inputs = frontiers
            .iter()
            .map(|frontier| {
                let [indices, counts, destinations] = frontier.indices();
                let words = [frontier.metadata(), indices, counts, destinations];
                let length = frontier.rows() as usize * frontier.model_dim();
                (
                    fixture.guarded(&frontier.current_qkv, sentinel),
                    words.map(|words| (fixture.guarded(&words, word), words)),
                    fixture.guarded(&vec![sentinel; length], sentinel),
                )
            })
            .collect::<Vec<_>>();
        fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
            (buffer, range.clone())
        }
        let mut encoding = fixture.encoding();
        for (frontier, (current_qkv, [metadata, indices, counts, destinations], output)) in
            frontiers.iter().zip(&inputs)
        {
            let buffers = [
                range(&prefix_kv),
                range(&node_kv),
                range(current_qkv),
                range(&cosines),
                range(&sines),
                range(&metadata.0),
                range(&indices.0),
                range(&counts.0),
                range(&destinations.0),
                range(output),
            ];
            // SAFETY: the guarded ranges hold every element the frontier indexes, which meets the preconditions.
            unsafe { frontier.encode(kernel, buffers, &mut encoding) };
        }
        KernelFixture::complete(encoding);
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            KernelFixture::assert_unchanged(&prefix_kv, sentinel, &first.prefix_kv, "prefix_kv");
            KernelFixture::assert_unchanged(&cosines, table, &first.cosines, "cosines");
            KernelFixture::assert_unchanged(&sines, table, &first.sines, "sines");
            let outputs = frontiers
                .iter()
                .zip(&inputs)
                .map(|(frontier, (current_qkv, words, output))| {
                    KernelFixture::assert_unchanged(current_qkv, sentinel, &frontier.current_qkv, "current_qkv");
                    for (guarded, payload) in words {
                        KernelFixture::assert_unchanged(guarded, word, payload, "AncestorAttention words");
                    }
                    KernelFixture::read_guarded(output, sentinel)
                })
                .collect();
            (outputs, KernelFixture::read_guarded(&node_kv, sentinel))
        }
    }

    pub fn label(&self) -> String {
        format!(
            "heads {} rows {} prefix {} counts {:?} stride {} capacity {}",
            self.num_heads,
            self.rows(),
            self.prefix_length,
            self.ancestors.iter().map(Vec::len).collect::<Vec<_>>(),
            self.ancestor_stride,
            self.node_capacity
        )
    }
}
