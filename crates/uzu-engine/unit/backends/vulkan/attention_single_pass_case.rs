use std::{fmt::Debug, ops::Range, sync::Arc};

use bytemuck::{AnyBitPattern, NoUninit};
use num_traits::Float;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{ring::RingParams, trie::TrieNode},
            kernel::AttentionSinglePassKernel,
        },
        cpu::Cpu,
        vulkan::{AttentionSinglePassVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// One raw AttentionSinglePass dispatch shared by the CPU and Vulkan runners and the FP64 oracle: FP32 values stored as
/// the kernel's type, queries `[heads, suffix, head_dim]`, keys and values of `prefix + suffix` rows at independent head
/// and row strides whose padding holds NaN, and the mask: a ring, a trie as preorder parent links, a window, causality.
#[derive(Clone)]
pub struct AttentionSinglePassCase {
    pub head_dim: u32,
    pub num_heads: u32,
    pub gqa_factor: u32,
    pub prefix_length: u32,
    pub suffix_length: u32,
    pub k_strides: (u32, u32),
    pub v_strides: (u32, u32),
    pub ring: Option<RingParams>,
    pub parents: Option<Vec<Option<u32>>>,
    pub window: Option<u32>,
    pub is_causal: bool,
    pub scale: f32,
    pub queries: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub sinks: Option<Vec<f32>>,
}

/// A hashed value in [-1, 1).
pub fn hashed(
    index: u32,
    seed: u32,
) -> f32 {
    let hash =
        (index ^ seed.wrapping_mul(0x27d4_eb2f)).wrapping_mul(0x9e37_79b9).rotate_left(13).wrapping_mul(0x85eb_ca6b);
    (hash >> 8) as f32 / (1 << 23) as f32 - 1.0
}

impl AttentionSinglePassCase {
    /// Causal attention with hashed values; keys are row-major with padded heads and rows, values head-major with
    /// padded rows, so the strides differ.
    pub fn new(
        head_dim: u32,
        num_heads: u32,
        gqa_factor: u32,
        prefix_length: u32,
        suffix_length: u32,
        seed: u32,
    ) -> Self {
        let (kv_heads, rows) = (num_heads / gqa_factor, prefix_length + suffix_length);
        let k_strides = (head_dim + 8, kv_heads * (head_dim + 8) + 16);
        let v_strides = (rows * (head_dim + 4), head_dim + 4);
        let cache = |(head_stride, row_stride): (u32, u32), seed: u32| {
            let mut cache = vec![f32::NAN; ((kv_heads - 1) * head_stride + rows.max(1) * row_stride) as usize];
            for (head, row, j) in itertools::iproduct!(0..kv_heads, 0..rows, 0..head_dim) {
                let index = head * head_stride + row * row_stride + j;
                cache[index as usize] = hashed(index, seed);
            }
            cache
        };
        Self {
            head_dim,
            num_heads,
            gqa_factor,
            prefix_length,
            suffix_length,
            k_strides,
            v_strides,
            ring: None,
            parents: None,
            window: None,
            is_causal: true,
            scale: 1.0 / (head_dim as f32).sqrt(),
            queries: (0..num_heads * suffix_length * head_dim).map(|index| hashed(index, seed)).collect(),
            keys: cache(k_strides, seed + 1),
            values: cache(v_strides, seed + 2),
            sinks: None,
        }
    }

    /// The model's cache layout: keys and values `[rows, kv_heads, head_dim]` without padding, so the head stride is
    /// `head_dim` and the row stride `kv_heads * head_dim`.
    pub fn with_model_layout(mut self) -> Self {
        let (kv_heads, dim) = (self.num_heads / self.gqa_factor, self.head_dim);
        let pack = |cache: &[f32], strides| -> Vec<f32> {
            itertools::iproduct!(0..self.sequence_length(), 0..kv_heads, 0..dim)
                .map(|(row, kv_head, j)| cache[self.index(strides, kv_head * self.gqa_factor, row, j)])
                .collect()
        };
        let (keys, values) = (pack(&self.keys, self.k_strides), pack(&self.values, self.v_strides));
        (self.keys, self.values, self.k_strides, self.v_strides) =
            (keys, values, (dim, kv_heads * dim), (dim, kv_heads * dim));
        self
    }

    pub fn sequence_length(&self) -> u32 {
        self.prefix_length + self.suffix_length
    }

    /// Index of element `j` of row `row` of the head serving query head `head` in a cache of `strides`.
    pub fn index(
        &self,
        (head_stride, row_stride): (u32, u32),
        head: u32,
        row: u32,
        j: u32,
    ) -> usize {
        (head / self.gqa_factor * head_stride + row * row_stride + j) as usize
    }

    /// A random preorder trie of the suffix: each node's parent is on the path to the previous node, or none.
    pub fn with_random_trie(
        mut self,
        seed: u32,
    ) -> Self {
        let mut path: Vec<u32> = Vec::new();
        let parents = (0..self.suffix_length)
            .map(|node| {
                let keep = ((hashed(node, seed) + 1.0) * 0.5 * (path.len() + 1) as f32) as usize;
                path.truncate(keep.min(path.len()));
                let parent = path.last().copied();
                path.push(node);
                parent
            })
            .collect();
        self.parents = Some(parents);
        self
    }

    /// `node`, its parent, and so on up to its root.
    fn path(
        parents: &[Option<u32>],
        node: u32,
    ) -> impl Iterator<Item = u32> {
        std::iter::successors(Some(node), |&node| parents[node as usize])
    }

    fn depth(
        parents: &[Option<u32>],
        node: u32,
    ) -> u32 {
        Self::path(parents, node).count() as u32 - 1
    }

    /// The canonical nodes of the trie: preorder index, last index of the subtree, depth.
    pub fn trie_nodes(&self) -> Option<Vec<TrieNode>> {
        Some(Self::nodes(self.parents.as_ref()?))
    }

    /// The canonical nodes of the trie of preorder parent links: preorder index, last index of the subtree, depth.
    pub fn nodes(parents: &[Option<u32>]) -> Vec<TrieNode> {
        let count = parents.len() as u32;
        (0..count)
            .map(|node| TrieNode {
                trie_start: node,
                trie_end: (node..count)
                    .take_while(|&other| Self::path(parents, other).any(|ancestor| ancestor == node))
                    .last()
                    .unwrap(),
                height: Self::depth(parents, node),
            })
            .collect()
    }

    /// Keys `query` attends to, in index order, from logical positions: a ring holds positions oldest first from its
    /// offset, filled ones only, and the suffix follows the filled positions; otherwise rows are positions. Suffix rows
    /// sit at their index or trie depth; causality admits earlier rows or trie ancestors, never limiting the prefix;
    /// the window compares logical positions.
    pub fn allowed(
        &self,
        query: u32,
    ) -> Vec<u32> {
        let mut positions: Vec<Option<u32>> = (0..self.prefix_length).map(Some).collect();
        let mut base = self.prefix_length;
        if let Some(ring) = self.ring {
            positions = vec![None; self.prefix_length as usize];
            for position in 0..ring.ring_length.min(self.prefix_length) {
                positions[((ring.ring_offset + position) % self.prefix_length) as usize] = Some(position);
            }
            base = ring.ring_length;
        }
        let suffix_position = |row: u32| base + self.parents.as_ref().map_or(row, |parents| Self::depth(parents, row));
        let query_position = suffix_position(query);
        let visible = |row: u32| match &self.parents {
            _ if !self.is_causal => true,
            Some(parents) => Self::path(parents, query).any(|ancestor| ancestor == row),
            None => row <= query,
        };
        positions.extend((0..self.suffix_length).map(|row| visible(row).then(|| suffix_position(row))));
        let in_window = |position: u32| match self.window {
            None => true,
            Some(window) if self.is_causal => position <= query_position && query_position - position < window,
            Some(window) => position.abs_diff(query_position) <= window / 2,
        };
        (0..self.sequence_length()).filter(|&key| positions[key as usize].is_some_and(in_window)).collect()
    }

    pub fn stored<T: ArrayElement + Float>(values: &[f32]) -> Vec<T> {
        values.iter().map(|&value| T::from(value).unwrap()).collect()
    }

    fn widened<T: ArrayElement + Float>(values: &[f32]) -> Vec<f64> {
        Self::stored::<T>(values).iter().map(|value| value.to_f64().unwrap()).collect()
    }

    /// Expected output `[suffix, heads, head_dim]` from FP64 arithmetic on the stored inputs, with the bound on the
    /// FP32 arithmetic error of the CPU and Vulkan kernels (storage rounding apart). `None` marks the elements whose
    /// class the CPU's ordered FP32 steps decide: a used NaN or infinite value or score, a NaN or +inf sink against used
    /// keys, or a first used key scoring -inf with no finite sink; they are compared with the CPU's class directly.
    /// Nothing visible gives NaN without a sink and 0 with one.
    pub fn oracle<T: ArrayElement + Float>(&self) -> Vec<Option<(f64, f64)>> {
        let (queries, keys, values) =
            (Self::widened::<T>(&self.queries), Self::widened::<T>(&self.keys), Self::widened::<T>(&self.values));
        let sinks = self.sinks.as_ref().map(|sinks| Self::widened::<T>(sinks));
        let (dim, scale) = (self.head_dim, f64::from(self.scale));
        let mut expected = Vec::with_capacity((self.suffix_length * self.num_heads * dim) as usize);
        for (query, head) in itertools::iproduct!(0..self.suffix_length, 0..self.num_heads) {
            let allowed = self.allowed(query);
            let sink = sinks.as_ref().map(|sinks| sinks[head as usize]);
            // Scores in FP64 with the sums of their terms' magnitudes.
            let scores = allowed
                .iter()
                .map(|&key| {
                    (0..dim).fold((0.0, 0.0), |(score, magnitude), j| {
                        let term = scale
                            * queries[((head * self.suffix_length + query) * dim + j) as usize]
                            * keys[self.index(self.k_strides, head, key, j)];
                        (score + term, magnitude + term.abs())
                    })
                })
                .collect::<Vec<_>>();
            let rows = allowed
                .iter()
                .map(|&key| (0..dim).map(|j| values[self.index(self.v_strides, head, key, j)]).collect())
                .collect::<Vec<Vec<f64>>>();
            let ordered = scores.iter().any(|&(score, _)| score.is_nan() || score == f64::INFINITY)
                || (!allowed.is_empty() && sink.is_some_and(|sink| sink.is_nan() || sink == f64::INFINITY))
                || (scores.first().is_some_and(|&(score, _)| score == f64::NEG_INFINITY)
                    && sink.is_none_or(|sink| sink == f64::NEG_INFINITY));
            if ordered || allowed.is_empty() {
                let value = (!ordered).then(|| (sink.map_or(f64::NAN, |_| 0.0), 0.0));
                expected.extend((0..dim).map(|_| value));
                continue;
            }
            let max = scores.iter().map(|&(score, _)| score).chain(sink).fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|&(score, _)| (score - max).exp()).collect::<Vec<_>>();
            for j in 0..dim as usize {
                let column = rows.iter().map(|row| row[j]).collect::<Vec<_>>();
                expected.push(column.iter().all(|value| value.is_finite()).then(|| {
                    Self::finite_bound(&scores, &weights, &column, sink.map(|sink| (sink - max).exp()), max, sink, dim)
                }));
            }
        }
        expected
    }

    /// The exact output o = Σ w v / W, w = e^(s - max), and the bound |ô - o| <= |N̂ - o L̂| / L̂ + quotient rounding
    /// for the computed numerator N̂ and normalizer L̂. With u = 2^-24 and γ_n = n u / (1 - n u) (γ'_n for FP64's 2^-53):
    /// - a score's D products of the scaled query (rounded once) are summed in any order: |ŝ - s| <= γ_(D+1) Σ|terms|,
    ///   plus 2D products and partial sums the device flushes below 2^-126, plus this FP64 score's own γ'_(D+2) Σ|terms|
    ///   (two products per term then D sums);
    /// - a weight passes through its own e^x and one rescale per later rise of the running maximum (R bounds the keys
    ///   that may rise above every earlier one within the score errors; a tile rises only if one of its keys does).
    ///   Vulkan's exp is within 3 + 2|x| ULPs (≤ 2^-23 relative each): 2u (3 + 2|x|). The subnormal path's exp2 of
    ///   z = x log2(e) + 24, |z| <= 1.443|x|, has 2u (3 + 2|z|) and an exponent off by |x| u (constant) + 1.443|x| u +
    ///   |z| u (two roundings), times ln 2 relative: u (6 + 5.8|x|) + 2.7 u|x|. So u (6 + 9|x|) per evaluation covers
    ///   both and the CPU's exp; one weight's evaluations have arguments summing to max - ŝ;
    /// - each product or sum of a weight or weighted value rounds once, relative u, or absolute 2^-150 onto the
    ///   subnormal grid: per term 1 product, its own sum, at most n - 1 later sums, R rescales and 15 slice sums (16
    ///   slices of a 1024-thread workgroup), n + R + 16; in all, at most 2n + R + 16 roundings per element;
    /// - a weight or rescale rounded onto the subnormal grid errs by 2^-150 on its own value or on the weighted values
    ///   it rescales, at most n of |v| <= max|v| each: n (R + 1) of them per element;
    /// - the quotient is within 2.5 ULPs, 5u, and flushed below 2^-126 at most.
    fn finite_bound(
        scores: &[(f64, f64)],
        weights: &[f64],
        column: &[f64],
        sink_weight: Option<f64>,
        max: f64,
        sink: Option<f64>,
        dim: u32,
    ) -> (f64, f64) {
        let (u, u64) = (2f64.powi(-24), 2f64.powi(-53));
        let gamma = |n: f64, u: f64| n * u / (1.0 - n * u);
        let (flushed, grid) = (2f64.powi(-126), 2f64.powi(-150));
        let (n, dim) = (scores.len() as f64, f64::from(dim));
        // A key scoring -inf has weight exactly 0 on every side and no error.
        let errors = scores
            .iter()
            .map(|&(score, magnitude)| match score.is_finite() {
                true => (gamma(dim + 1.0, u) + gamma(dim + 2.0, u64)) * magnitude + 2.0 * dim * flushed,
                false => 0.0,
            })
            .collect::<Vec<_>>();
        let error_max = errors.iter().copied().fold(0.0, f64::max);
        let mut running = sink.unwrap_or(f64::NEG_INFINITY);
        let mut rises = 0.0;
        for (&(score, _), &error) in scores.iter().zip(&errors) {
            if score + error > running {
                rises += 1.0;
            }
            running = running.max(score - error);
        }
        let evaluations = |x: f64, count: f64| {
            let epsilon = u * (6.0 * count + 9.0 * x);
            epsilon / (1.0 - epsilon)
        };
        let relative = scores
            .iter()
            .zip(&errors)
            .map(|(&(score, _), &error)| match score.is_finite() {
                true => (error + evaluations(max - score + error + error_max, rises + 1.0)).exp_m1(),
                false => 0.0,
            })
            .collect::<Vec<_>>();
        let sink_relative = sink
            .filter(|sink| sink.is_finite())
            .map_or(0.0, |sink| evaluations(max - sink + error_max, rises).exp_m1());
        let g = gamma(n + rises + 16.0, u);
        let normalizer = weights.iter().sum::<f64>() + sink_weight.unwrap_or(0.0);
        let output = weights.iter().zip(column).map(|(weight, value)| weight * value).sum::<f64>() / normalizer;
        let deviation = weights
            .iter()
            .zip(column)
            .zip(&relative)
            .map(|((weight, value), r)| {
                weight * (r * (value - output).abs() + g * (1.0 + r) * (value.abs() + output.abs()))
            })
            .sum::<f64>()
            + sink_weight.unwrap_or(0.0) * (sink_relative + g * (1.0 + sink_relative)) * output.abs();
        // L̂ >= Σ w (1 - r) (1 - g) in the exact weights' units, which the computed weights match within e^(errors).
        let lowest = weights.iter().zip(&relative).map(|(weight, r)| weight * r).sum::<f64>()
            + sink_weight.unwrap_or(0.0) * sink_relative;
        let denominator = (normalizer - lowest) * (1.0 - g);
        let units = (-(error_max + evaluations(2.0 * error_max, rises + 1.0))).exp();
        let largest = column.iter().fold(0.0f64, |largest, value| largest.max(value.abs()));
        let absolute = (n * (rises + 1.0) * grid * (largest + output.abs()) + (2.0 * n + rises + 16.0) * grid)
            / (units * denominator);
        // FP64 weights, sums and quotient of this oracle: within γ'_(2n + 4) of their magnitudes.
        let oracle = gamma(2.0 * n + 4.0, u64) * (output.abs() + largest);
        let arithmetic = deviation / denominator + absolute + oracle;
        (output, arithmetic + 5.0 * u * (output.abs() + arithmetic) + flushed)
    }

    /// The CPU kernel through the shared trait; trie nodes travel as their words.
    pub fn cpu<T: ArrayElement + Float>(&self) -> Vec<T> {
        let context = create_context::<Cpu>();
        let kernel = <<Cpu as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
            &context,
            T::data_type(),
            self.head_dim,
            self.sinks.is_some(),
            self.ring.is_some(),
            self.is_causal,
            self.parents.is_some(),
            self.window.is_some(),
        )
        .expect("CPU AttentionSinglePass");
        let buffer = |values: &[f32]| create_buffer_with_data::<Cpu, T>(&context, &Self::stored::<T>(values));
        let (queries, keys, values) = (buffer(&self.queries), buffer(&self.keys), buffer(&self.values));
        let sinks = self.sinks.as_ref().map(|sinks| buffer(sinks));
        let trie =
            self.trie_nodes().map(|nodes| create_buffer_with_data::<Cpu, u32>(&context, bytemuck::cast_slice(&nodes)));
        let mut out = buffer(&vec![0.0; (self.suffix_length * self.num_heads * self.head_dim).max(1) as usize]);
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        kernel.encode(
            &queries,
            &keys,
            &values,
            &mut out,
            self.gqa_factor,
            self.sequence_length(),
            self.k_strides.0,
            self.k_strides.1,
            self.v_strides.0,
            self.v_strides.1,
            self.ring,
            self.scale,
            trie.as_ref(),
            self.window,
            sinks.as_ref(),
            self.num_heads,
            self.suffix_length,
            &mut command_buffer,
        );
        submit_command_buffer(command_buffer);
        buffer_to_vec::<Cpu, T>(&out)
    }

    pub fn vulkan_kernel<T: ArrayElement>(
        &self,
        fixture: &KernelFixture,
    ) -> AttentionSinglePassVulkanKernel {
        AttentionSinglePassVulkanKernel::new(
            &fixture.context,
            T::data_type(),
            self.head_dim,
            self.sinks.is_some(),
            self.ring.is_some(),
            self.is_causal,
            self.parents.is_some(),
            self.window.is_some(),
        )
        .expect("Vulkan AttentionSinglePass")
    }

    /// Records the case's dispatch over `[queries, keys, values, out]`, the trie words and the sinks.
    ///
    /// # Safety
    /// The buffers hold every row at the case's strides, every head and query, the trie every suffix row and the
    /// sinks every head, aligned; the output aliases nothing.
    pub unsafe fn encode(
        &self,
        kernel: &AttentionSinglePassVulkanKernel,
        [queries, keys, values, out]: [(&Arc<VkBuffer>, Range<u64>); 4],
        trie: Option<(&Arc<VkBuffer>, Range<u64>)>,
        sinks: Option<(&Arc<VkBuffer>, Range<u64>)>,
        encoding: &mut VkCommandBufferEncoding,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            kernel.encode(
                queries,
                keys,
                values,
                out,
                self.gqa_factor,
                self.sequence_length(),
                self.k_strides.0,
                self.k_strides.1,
                self.v_strides.0,
                self.v_strides.1,
                self.ring,
                self.scale,
                trie,
                self.window,
                sinks,
                self.num_heads,
                self.suffix_length,
                encoding,
            )
        }
    }

    /// One dispatch over guarded buffers; returns the output after asserting every guard and every input unchanged.
    pub fn gpu<T: ArrayElement + Float + NoUninit + AnyBitPattern>(
        &self,
        fixture: &KernelFixture,
        kernel: &AttentionSinglePassVulkanKernel,
    ) -> Vec<T> {
        let sentinel = T::from(-777.0).unwrap();
        let guarded =
            |values: &[f32]| (Self::stored::<T>(values), fixture.guarded(&Self::stored::<T>(values), sentinel));
        let inputs = [&self.queries, &self.keys, &self.values].map(|values| guarded(values));
        let sinks = self.sinks.as_ref().map(|sinks| guarded(sinks));
        let words = self.trie_nodes().map(|nodes| bytemuck::cast_slice::<TrieNode, u32>(&nodes).to_vec());
        let trie = words.as_ref().map(|words| fixture.guarded(words, u32::MAX));
        let out =
            fixture.guarded(&vec![sentinel; (self.suffix_length * self.num_heads * self.head_dim) as usize], sentinel);
        fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
            (buffer, range.clone())
        }
        let mut encoding = fixture.encoding();
        let buffers = [range(&inputs[0].1), range(&inputs[1].1), range(&inputs[2].1), range(&out)];
        // SAFETY: the guarded buffers hold every row, head and query of the case; the output aliases nothing.
        unsafe {
            self.encode(
                kernel,
                buffers,
                trie.as_ref().map(range),
                sinks.as_ref().map(|(_, sinks)| range(sinks)),
                &mut encoding,
            )
        };
        KernelFixture::complete(encoding);
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            for (payload, guarded) in inputs.iter().chain(&sinks) {
                KernelFixture::assert_unchanged(guarded, sentinel, payload, "AttentionSinglePass input");
            }
            if let (Some(words), Some(trie)) = (&words, &trie) {
                KernelFixture::assert_unchanged(trie, u32::MAX, words, "trie");
            }
            KernelFixture::read_guarded(&out, sentinel)
        }
    }

    /// Compares `actual` with the oracle: finite values within the arithmetic bound plus half a storage spacing, the
    /// oracle's NaN exactly, and elements of ordered class nonfinite with `cpu`'s class when given (the CPU's own run
    /// passes none). Returns the max absolute error, max error over bound, nonfinite count and violations.
    pub fn compare<T: ArrayElement + Float + Debug>(
        expected: &[Option<(f64, f64)>],
        actual: &[T],
        cpu: Option<&[T]>,
        label: &str,
    ) -> [f64; 4] {
        assert_eq!(expected.len(), actual.len(), "{label}: length");
        let mut errors = [0.0f64; 4];
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let actual = actual.to_f64().unwrap();
            let within = match *expected {
                None => {
                    errors[2] += 1.0;
                    let class = cpu.map_or(actual, |cpu| cpu[index].to_f64().unwrap());
                    !class.is_finite() && (class == actual || (class.is_nan() && actual.is_nan()))
                },
                Some((value, _)) if value.is_nan() => {
                    errors[2] += 1.0;
                    actual.is_nan()
                },
                Some((value, bound)) => {
                    // Half the spacing of the storage type around the larger magnitude, on the subnormal grid below.
                    let (epsilon, smallest) =
                        (T::epsilon().to_f64().unwrap(), T::min_positive_value().to_f64().unwrap());
                    let magnitude = (value.abs() + bound).max(smallest);
                    let spacing = (epsilon * 2f64.powf(magnitude.log2().floor())).max(smallest * epsilon);
                    let allowed = bound
                        + if size_of::<T>() == 4 {
                            0.0
                        } else {
                            spacing / 2.0
                        };
                    let error = (actual - value).abs();
                    errors[0] = errors[0].max(error);
                    errors[1] = errors[1].max(error / allowed.max(f64::MIN_POSITIVE));
                    error <= allowed
                },
            };
            if !within {
                if errors[3] == 0.0 {
                    eprintln!("{label}: element {index}: expected {expected:?}, got {actual:e}");
                }
                errors[3] += 1.0;
            }
        }
        errors
    }

    pub fn label(&self) -> String {
        format!(
            "D {} heads {}/{} prefix {} suffix {} ring {:?} trie {} window {:?} causal {} sinks {}",
            self.head_dim,
            self.num_heads,
            self.gqa_factor,
            self.prefix_length,
            self.suffix_length,
            self.ring.map(|ring| (ring.ring_offset, ring.ring_length)),
            self.parents.is_some(),
            self.window,
            self.is_causal,
            self.sinks.is_some()
        )
    }
}
