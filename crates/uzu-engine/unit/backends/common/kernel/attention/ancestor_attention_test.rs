#[cfg(backend = "metal")]
use std::time::Instant;
use std::{sync::Arc, time::Duration};

use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

#[cfg(backend = "metal")]
use crate::backends::metal::Metal;
use crate::{
    backends::{
        common::{
            Backend, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::weaver::MetadataIdx,
            kernel::{AncestorAttentionKernel, AttentionPrepareKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCache,
    tests::{
        assert::assert_eq_float,
        attention::kv_layout,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
    },
};

const HEAD_DIM: usize = 128;
const NUM_HEADS: u32 = 16;
const MAX_DEPTH: u32 = 16;

struct Runner<B: Backend> {
    context: Arc<B::Context>,
    kernel: <B::Kernels as Kernels>::AncestorAttentionKernel,
    prefix_kv: KVCache<B>,
    node_kv: KVCache<B>,
    current_qkv: B::GlobalGpuBuffer,
    rope_inverse_frequencies: B::GlobalGpuBuffer,
    node_metadata: B::GlobalGpuBuffer,
    ancestor_indices: B::GlobalGpuBuffer,
    ancestor_counts: B::GlobalGpuBuffer,
    node_indices: B::GlobalGpuBuffer,
    output: B::GlobalGpuBuffer,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    node_capacity: u32,
}

impl<B: Backend> Runner<B> {
    fn new(
        rows: usize,
        prefix_length: usize,
        ancestor_stride: usize,
        node_capacity: usize,
        layout_override: Option<u32>,
    ) -> Self {
        assert!(rows < node_capacity && ancestor_stride > 0);
        let model_dim = NUM_HEADS as usize * HEAD_DIM;
        let qkv_width = 3 * model_dim;
        let kv_width = 2 * model_dim;
        let values = |length, offset| {
            (0..length)
                .map(|index| bf16::from_f32((((index + offset) * 17 % 251) as f32 - 125.0) / 128.0))
                .collect::<Vec<_>>()
        };
        let first_output_node = node_capacity - rows;
        let node_indices = (first_output_node..node_capacity).map(|node| node as u32).collect::<Vec<_>>();
        let ancestor_counts =
            (0..rows).map(|row| [0u32, 1, 4, 15][row % 4].min(ancestor_stride as u32)).collect::<Vec<_>>();
        let mut ancestor_indices = vec![0; rows * ancestor_stride];
        for row in 0..rows {
            for offset in 0..ancestor_counts[row] as usize {
                ancestor_indices[row * ancestor_stride + offset] =
                    ((row * ancestor_stride + offset) % first_output_node) as u32;
            }
        }

        let rope_inverse_frequencies =
            (0..HEAD_DIM / 2).map(|pair| 1.0 / 10_000f32.powf(2.0 * pair as f32 / HEAD_DIM as f32)).collect::<Vec<_>>();
        let mut node_metadata = vec![0u32; rows * MetadataIdx::COUNT];
        for row in 0..rows {
            node_metadata[MetadataIdx::Depth as usize * rows + row] = ancestor_counts[row];
        }

        let context = create_context::<B>();
        let mut layout = kv_layout::<B>(&context, HEAD_DIM as u32, NUM_HEADS);
        if let Some(layout_override) = layout_override {
            layout.backend_opaque_layout = layout_override;
        }
        let kernel = <B::Kernels as Kernels>::AncestorAttentionKernel::new(
            context.as_ref(),
            HEAD_DIM as u32,
            NUM_HEADS,
            layout.backend_opaque_layout,
        )
        .unwrap();
        let prepare_cache = |length: usize, offset| {
            let mut command = context.create_command_buffer(None, None).unwrap();
            let mut kv = KVCache::<B>::allocate_scratch(layout, length as u32, &mut command).unwrap();
            let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
                &context,
                DataType::BF16,
                HEAD_DIM as u32,
                false,
                false,
                NUM_HEADS,
                layout.backend_opaque_layout,
                true,
                false,
                false,
                false,
                false,
                0,
                false,
            )
            .unwrap();
            let data = create_buffer_with_data::<B, bf16>(&context, &values(length * kv_width, offset));
            command.encode_fill(kv.k_codes.as_mut(), 0);
            command.encode_fill(kv.v_codes.as_mut(), 128);
            command.encode_fill(kv.k_scales.as_mut(), 0);
            command.encode_fill(kv.v_scales.as_mut(), 0);
            let mut unused_q_codes = create_buffer::<B, i8>(&context, 1);
            let mut unused_q_scales = create_buffer::<B, bf16>(&context, 1);
            prepare.encode(
                &data,
                &mut unused_q_codes,
                &mut unused_q_scales,
                Some(kv.k_codes.as_mut()),
                Some(kv.k_scales.as_mut()),
                Some(kv.v_codes.as_mut()),
                Some(kv.v_scales.as_mut()),
                None::<&B::GlobalGpuBuffer>,
                None::<&B::GlobalGpuBuffer>,
                None::<&B::GlobalGpuBuffer>,
                None::<&B::GlobalGpuBuffer>,
                None::<&B::GlobalGpuBuffer>,
                None::<&B::GlobalGpuBuffer>,
                None::<&mut B::GlobalGpuBuffer>,
                Default::default(),
                0,
                None,
                None,
                None,
                None,
                0,
                kv_width as u32,
                length as u32,
                0,
                None,
                None::<&B::GlobalGpuBuffer>,
                &mut command,
            );
            block_on(command.end_encoding().submit().wait_until_completed()).unwrap();
            kv
        };
        Self {
            prefix_kv: prepare_cache(prefix_length, 0),
            node_kv: prepare_cache(node_capacity, 11),
            current_qkv: create_buffer_with_data::<B, bf16>(&context, &values(rows * qkv_width, 29)),
            rope_inverse_frequencies: create_buffer_with_data::<B, f32>(&context, &rope_inverse_frequencies),
            node_metadata: create_buffer_with_data::<B, u32>(&context, &node_metadata),
            ancestor_indices: create_buffer_with_data::<B, u32>(&context, &ancestor_indices),
            ancestor_counts: create_buffer_with_data::<B, u32>(&context, &ancestor_counts),
            node_indices: create_buffer_with_data::<B, u32>(&context, &node_indices),
            output: create_buffer::<B, bf16>(&context, rows * NUM_HEADS as usize * HEAD_DIM),
            context,
            kernel,
            rows: rows as u32,
            prefix_length: prefix_length as u32,
            ancestor_stride: ancestor_stride as u32,
            node_capacity: node_capacity as u32,
        }
    }

    fn encode(
        &mut self,
        repetitions: u32,
    ) -> Option<Duration> {
        let mut command_buffer = self.context.create_command_buffer(None, None).unwrap();
        for _ in 0..repetitions {
            self.kernel.encode(
                self.prefix_kv.k_codes.as_ref(),
                self.prefix_kv.k_scales.as_ref(),
                self.prefix_kv.v_codes.as_ref(),
                self.prefix_kv.v_scales.as_ref(),
                self.node_kv.k_codes.as_mut(),
                self.node_kv.k_scales.as_mut(),
                self.node_kv.v_codes.as_mut(),
                self.node_kv.v_scales.as_mut(),
                &self.current_qkv,
                &self.rope_inverse_frequencies,
                &self.node_metadata,
                &self.ancestor_indices,
                &self.ancestor_counts,
                &self.node_indices,
                &mut self.output,
                self.rows,
                self.prefix_length,
                self.ancestor_stride,
                self.node_capacity,
                MAX_DEPTH,
                1.0,
                1.0 / (HEAD_DIM as f32).sqrt(),
                &mut command_buffer,
            );
        }
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap().gpu_execution_time()
    }
}

#[uzu_test]
fn ancestor_attention_matches_cpu() {
    #[cfg(backend = "metal")]
    let layouts = [crate::backends::metal::KVLayout::RowMajor, crate::backends::metal::KVLayout::MxuTiled]
        .map(|layout| Some(layout as u32));
    #[cfg(not(backend = "metal"))]
    let layouts = [None];

    for_each_non_cpu_backend!(|B| {
        for (rows, prefix) in [(1, 8), (4, 16), (4, 33)] {
            let mut cpu = Runner::<Cpu>::new(rows, prefix, 16, 65, None);
            cpu.encode(1);
            let expected = buffer_to_vec::<Cpu, bf16>(&cpu.context, &cpu.output);

            for layout in layouts {
                let mut runner = Runner::<B>::new(rows, prefix, 16, 65, layout);
                runner.encode(1);
                let actual = buffer_to_vec::<B, bf16>(&runner.context, &runner.output);
                assert_eq_float(&expected, &actual, 0.02, "AncestorAttention output");
            }
        }
    });
}

#[cfg(backend = "metal")]
#[uzu_test]
#[ignore]
fn benchmark_ancestor_attention() {
    const BATCH: u32 = 32;
    const SAMPLES: usize = 50;

    let mut runner = Runner::<Metal>::new(8, 16, 8, 65, None);
    let mut run = || runner.encode(BATCH).unwrap().div_f64(BATCH as f64);
    let warmup = Instant::now();
    while warmup.elapsed() < Duration::from_millis(500) {
        run();
    }
    let mut samples = (0..SAMPLES).map(|_| run()).collect::<Vec<_>>();
    samples.sort_unstable();
    eprintln!("ancestor_attention gpu={:?}", samples[SAMPLES / 2]);
}
