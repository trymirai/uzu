use std::{ops::Range, sync::Arc};

use half::bf16;

use super::{kernel_fixture::KernelFixture, round32};
use crate::{
    backends::{
        common::{Backend, Context, Kernels, kernel::AttentionPrepareKernel},
        cpu::Cpu,
        vulkan::{AttentionPrepareVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// Query elements and cache tokens past the ones a dispatch writes, which must keep their initial values.
const TAIL: usize = 3;

/// One raw AttentionPrepare dispatch shared by the CPU and Vulkan runners and the staged oracle: packed BF16 rows with
/// a trailing gate and padding, optional FP32 rotary tables of `batch_dim` rows of `rope_dim`, and the dimensions. With
/// KV the caches hold `kv_token_offset` earlier tokens before the batch and `TAIL` after it.
#[derive(Clone)]
pub struct AttentionPrepareCase {
    pub qkvg: Vec<bf16>,
    pub cosines: Option<Vec<f32>>,
    pub sines: Option<Vec<f32>>,
    pub num_q_heads: u32,
    pub num_kv_heads: Option<u32>,
    pub head_dim: u32,
    pub rope_dim: Option<u32>,
    pub kv_token_offset: Option<u32>,
    pub input_row_stride: u32,
    pub batch_dim: u32,
}

impl AttentionPrepareCase {
    /// Rows of arbitrary-mantissa values in [-4.2, 4.2] followed by a gate as wide as the queries and 3 padding elements,
    /// tables in (-1, 1) and, with KV, 2 earlier cache tokens; `seed` varies the patterns.
    pub fn new(
        num_q_heads: u32,
        num_kv_heads: Option<u32>,
        head_dim: u32,
        rope_dim: Option<u32>,
        batch_dim: u32,
        seed: usize,
    ) -> Self {
        let heads = num_q_heads + 2 * num_kv_heads.unwrap_or(0);
        let input_row_stride = (heads + num_q_heads) * head_dim + 3;
        let qkvg = (0..(batch_dim * input_row_stride) as usize)
            .map(|i| bf16::from_f32(((i * 37 + seed) % 509) as f32 / 61.0 - 4.17))
            .collect();
        let table = |offset: usize| {
            rope_dim.map(|rope_dim| {
                let length = (batch_dim * rope_dim) as usize;
                (0..length).map(|i| (((i * 53 + seed + offset) % 997) as f32 / 498.5 - 1.0) * 0.999_93).collect()
            })
        };
        Self {
            qkvg,
            cosines: table(0),
            sines: table(331),
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            kv_token_offset: num_kv_heads.map(|_| 2),
            input_row_stride,
            batch_dim,
        }
    }

    /// The initial `[queries, keys, values]`: distinct patterns, the caches empty without KV.
    pub fn initial(&self) -> [Vec<bf16>; 3] {
        let queries = (self.num_q_heads * self.batch_dim * self.head_dim) as usize + TAIL;
        let cache = self.num_kv_heads.map_or(0, |kv_heads| {
            (self.kv_token_offset.unwrap_or(0) + self.batch_dim) as usize * (kv_heads * self.head_dim) as usize
                + TAIL * (kv_heads * self.head_dim) as usize
        });
        let pattern =
            |length: usize, base: u16| (0..length).map(|i| bf16::from_bits(base + (i % 509) as u16)).collect();
        [pattern(queries, 0x3c00), pattern(cache, 0x4100), pattern(cache, 0xc100)]
    }

    /// The expected `[queries, keys, values]`, each element flagged when it is rotated, so NaN compares by class there:
    /// copies keep their bits. A rotated element is the FP64 staging of the CPU apply_rope: each product and the sum
    /// rounded to FP32 (subnormals kept), the signed pair negated exactly, then rounded to BF16.
    pub fn oracle(&self) -> [Vec<(bf16, bool)>; 3] {
        let mut outputs =
            self.initial().map(|values| values.into_iter().map(|value| (value, false)).collect::<Vec<_>>());
        let (q_heads, kv_heads) = (self.num_q_heads as usize, self.num_kv_heads.unwrap_or(0) as usize);
        let (head_dim, rope_dim) = (self.head_dim as usize, self.rope_dim.unwrap_or(0) as usize);
        let (batch_dim, offset) = (self.batch_dim as usize, self.kv_token_offset.unwrap_or(0) as usize);
        for batch in 0..batch_dim {
            for head in 0..q_heads + 2 * kv_heads {
                let row = batch * self.input_row_stride as usize + head * head_dim;
                for element in 0..head_dim {
                    let rotated = element < rope_dim && head < q_heads + kv_heads;
                    let value = match rotated {
                        true => {
                            let (half, table) = (rope_dim / 2, batch * rope_dim + element);
                            let first = element < half;
                            let paired = self.qkvg[row
                                + if first {
                                    element + half
                                } else {
                                    element - half
                                }]
                            .to_f64();
                            let signed_paired = if first {
                                -paired
                            } else {
                                paired
                            };
                            let cosine = f64::from(self.cosines.as_ref().unwrap()[table]);
                            let sine = f64::from(self.sines.as_ref().unwrap()[table]);
                            let input = self.qkvg[row + element].to_f64();
                            bf16::from_f32((round32(input * cosine) + round32(signed_paired * sine)) as f32)
                        },
                        false => self.qkvg[row + element],
                    };
                    let (output, index) = match head < q_heads {
                        true => (0, (head * batch_dim + batch) * head_dim + element),
                        false => {
                            let (kind, kv_head) = ((head - q_heads) / kv_heads, (head - q_heads) % kv_heads);
                            (1 + kind, ((offset + batch) * kv_heads + kv_head) * head_dim + element)
                        },
                    };
                    outputs[output][index] = (value, rotated);
                }
            }
        }
        outputs
    }

    /// The CPU kernel through the shared trait from the initial outputs; CPU buffers cannot be empty, and an empty batch
    /// writes nothing.
    pub fn cpu(&self) -> [Vec<bf16>; 3] {
        let [queries, keys, values] = self.initial();
        if self.batch_dim == 0 {
            return [queries, keys, values];
        }
        let context = create_context::<Cpu>();
        let kernel = <<Cpu as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
            &context,
            DataType::BF16,
            DataType::F32,
            self.num_kv_heads.is_some(),
            self.rope_dim.is_some(),
        )
        .expect("CPU AttentionPrepare");
        let qkvg = create_buffer_with_data::<Cpu, bf16>(&context, &self.qkvg);
        let mut query_buffer = create_buffer_with_data::<Cpu, bf16>(&context, &queries);
        let cache = |values: &[bf16]| self.num_kv_heads.map(|_| create_buffer_with_data::<Cpu, bf16>(&context, values));
        let (mut key_buffer, mut value_buffer) = (cache(&keys), cache(&values));
        let table =
            |table: &Option<Vec<f32>>| table.as_ref().map(|table| create_buffer_with_data::<Cpu, f32>(&context, table));
        let (cosines, sines) = (table(&self.cosines), table(&self.sines));
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        kernel.encode(
            &qkvg,
            &mut query_buffer,
            key_buffer.as_mut(),
            value_buffer.as_mut(),
            cosines.as_ref(),
            sines.as_ref(),
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rope_dim,
            self.kv_token_offset,
            self.input_row_stride,
            self.batch_dim,
            &mut command_buffer,
        );
        submit_command_buffer(command_buffer);
        let read = |buffer: Option<_>, initial: Vec<bf16>| {
            buffer.map_or(initial, |buffer| buffer_to_vec::<Cpu, bf16>(&buffer))
        };
        [buffer_to_vec::<Cpu, bf16>(&query_buffer), read(key_buffer, keys), read(value_buffer, values)]
    }

    pub fn vulkan_kernel(
        &self,
        fixture: &KernelFixture,
    ) -> AttentionPrepareVulkanKernel {
        AttentionPrepareVulkanKernel::new(
            &fixture.context,
            DataType::BF16,
            DataType::F32,
            self.num_kv_heads.is_some(),
            self.rope_dim.is_some(),
        )
        .expect("Vulkan AttentionPrepare")
    }

    /// Records every case into `encoding` over guarded ranges from the initial outputs and completes it. Returns each
    /// case's `[queries, keys, values]` after asserting every guard and the rows and tables unchanged.
    pub fn gpu(
        fixture: &KernelFixture,
        cases: &[Self],
        mut encoding: VkCommandBufferEncoding,
    ) -> Vec<[Vec<bf16>; 3]> {
        let (sentinel, table_sentinel) = (bf16::from_bits(0x7bad), -7.0f32);
        let kernels = cases.iter().map(|case| case.vulkan_kernel(fixture)).collect::<Vec<_>>();
        let buffers = cases
            .iter()
            .map(|case| {
                let table =
                    |table: &Option<Vec<f32>>| table.as_ref().map(|table| fixture.guarded(table, table_sentinel));
                let outputs = case.initial().map(|values| fixture.guarded(&values, sentinel));
                (fixture.guarded(&case.qkvg, sentinel), table(&case.cosines), table(&case.sines), outputs)
            })
            .collect::<Vec<_>>();
        fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
            (buffer, range.clone())
        }
        for ((case, kernel), (qkvg, cosines, sines, [queries, keys, values])) in
            cases.iter().zip(&kernels).zip(&buffers)
        {
            let kv = |guarded| case.num_kv_heads.map(|_| range(guarded));
            // SAFETY: the rows hold `batch_dim` rows of `input_row_stride` and the tables `batch_dim` rows of
            // `rope_dim`; queries hold every query head's tokens and the caches every written token, all aligned BF16
            // or FP32; the outputs alias nothing.
            unsafe {
                kernel.encode(
                    range(qkvg),
                    range(queries),
                    kv(keys),
                    kv(values),
                    cosines.as_ref().map(range),
                    sines.as_ref().map(range),
                    case.num_q_heads,
                    case.num_kv_heads,
                    case.head_dim,
                    case.rope_dim,
                    case.kv_token_offset,
                    case.input_row_stride,
                    case.batch_dim,
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
        cases
            .iter()
            .zip(&buffers)
            .map(|(case, (qkvg, cosines, sines, outputs))| {
                // SAFETY: the only command buffer using these buffers has completed.
                unsafe {
                    KernelFixture::assert_unchanged(qkvg, sentinel, &case.qkvg, "qkvg");
                    for (guarded, payload) in [(cosines, &case.cosines), (sines, &case.sines)] {
                        if let (Some(guarded), Some(payload)) = (guarded, payload) {
                            KernelFixture::assert_unchanged(guarded, table_sentinel, payload, "rotary table");
                        }
                    }
                    outputs.each_ref().map(|output| KernelFixture::read_guarded(output, sentinel))
                }
            })
            .collect()
    }

    pub fn label(&self) -> String {
        format!(
            "AttentionPrepare q {} kv {:?} head_dim {} rope {:?} batch {} stride {}",
            self.num_q_heads, self.num_kv_heads, self.head_dim, self.rope_dim, self.batch_dim, self.input_row_stride
        )
    }
}
