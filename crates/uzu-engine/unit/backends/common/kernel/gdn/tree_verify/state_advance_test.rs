#[cfg(backend = "metal")]
use std::time::Duration;

#[cfg(backend = "metal")]
use criterion::Criterion;
use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
#[cfg(backend = "metal")]
use uzu_engine_macros::uzu_bench;
use uzu_engine_macros::uzu_test;

#[cfg(backend = "metal")]
use crate::{
    backends::metal::Metal,
    tests::{cold_pool::ColdPool, helpers::buffer_size_bytes, matmul::iter_encode_loop_named},
};
use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            Kernels,
            kernel::{StateAdvanceKernel, delta_net_tree_verify::DeltaNetTreeVerify},
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::mixer::delta_net::{
        accepted_path::AcceptedPath,
        tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
    },
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

const TREE_SIZE: usize = 128;
const HEAD_DIM: usize = 128;
const PATH: &[u32] = &[0, 1, 3, 7, 15, 31, 47, 63, 79, 95, 111, 119, 123, 125, 126, 127];
const TEST_NUM_V_HEADS: usize = 6;
const TEST_NUM_K_HEADS: usize = 2;
const VERIFY_CONV_DIM: usize = 5;
const VERIFY_PROJ_DIM: usize = VERIFY_CONV_DIM + TEST_NUM_V_HEADS * HEAD_DIM + 3;

fn run<B: Backend, T: DataTypeRepr + Float>(accepted_indices: &[u32]) -> Vec<f32> {
    let context = block_on(B::Context::new()).expect("context");
    let kernel = <<B as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(
        &context,
        T::data_type(),
        HEAD_DIM as u32,
        TEST_NUM_V_HEADS as u32,
        TEST_NUM_K_HEADS as u32,
    )
    .expect("kernel");
    let key_len = TREE_SIZE * TEST_NUM_K_HEADS * HEAD_DIM;
    let value_len = TREE_SIZE * TEST_NUM_V_HEADS * HEAD_DIM;
    let scalar_len = TREE_SIZE * TEST_NUM_V_HEADS;
    let state_len = TEST_NUM_V_HEADS * HEAD_DIM * HEAD_DIM;
    let k_norm = (0..key_len).map(|i| T::from((i % 17) as f32 * 0.002 - 0.016).unwrap()).collect::<Vec<_>>();
    let v = (0..value_len).map(|i| T::from((i % 19) as f32 * 0.003 - 0.027).unwrap()).collect::<Vec<_>>();
    let log_decay = (0..scalar_len).map(|i| -0.01 - (i % 7) as f32 * 0.002).collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.2 + (i % 5) as f32 * 0.03).collect::<Vec<_>>();
    let initial_state = (0..state_len).map(|i| (i % 23) as f32 * 0.001 - 0.011).collect::<Vec<_>>();

    let k_norm = create_buffer_with_data::<B, T>(&context, &k_norm);
    let v = create_buffer_with_data::<B, T>(&context, &v);
    let log_decay = create_buffer_with_data::<B, f32>(&context, &log_decay);
    let beta = create_buffer_with_data::<B, f32>(&context, &beta);
    let accepted_len = accepted_indices.len();
    let accepted_indices = create_buffer_with_data::<B, u32>(&context, accepted_indices);
    let mut committed_state = create_buffer_with_data::<B, f32>(&context, &initial_state);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &k_norm,
        &v,
        &log_decay,
        &beta,
        &accepted_indices,
        &mut committed_state,
        accepted_len as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec(&*context, &committed_state)
}

#[uzu_test]
fn test_state_advance() {
    for path in [&PATH[..1], &PATH[..8], PATH] {
        for_each_non_cpu_backend!(|B| {
            assert_eq_float(&run::<Cpu, f32>(path), &run::<B, f32>(path), 2e-5, "F32 state");
            assert_eq_float(&run::<Cpu, bf16>(path), &run::<B, bf16>(path), 2e-5, "BF16 state");
        });
    }
}

/// `data` in scratch of `command_buffer`, copied from a buffer that `sources` keeps.
fn scratch<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    sources: &mut Vec<B::GlobalGpuBuffer>,
    data: &[T],
) -> B::ScratchBuffer {
    let source = create_buffer_with_data::<B, T>(context, data);
    let mut buffer = command_buffer.allocate_scratch(size_of_val(data)).expect("scratch");
    command_buffer.encode_copy(&source, &mut buffer);
    sources.push(source);
    buffer
}

/// The values of a caterpillar tree of `nodes` and the state after its verification, from a state advanced along
/// `indices` of another tree: by StateAdvance before the verification, or by the verification itself when `fused`.
fn advance_and_verify<B: Backend>(
    indices: &[u32],
    nodes: usize,
    fused: bool,
) -> (Vec<bf16>, Vec<f32>) {
    let context = block_on(B::Context::new()).expect("context");
    let arguments = TreeVerifyNewArguments {
        data_type: DataType::BF16,
        num_k_heads: TEST_NUM_K_HEADS as u32,
        num_v_heads: TEST_NUM_V_HEADS as u32,
        head_k_dim: HEAD_DIM as u32,
        head_v_dim: HEAD_DIM as u32,
    };
    let tree_verify = <B::Kernels as Kernels>::DeltaNetTreeVerify::new(&context, &arguments).expect("tree verify");
    let state_advance = <B::Kernels as Kernels>::StateAdvanceKernel::new(
        &context,
        DataType::BF16,
        HEAD_DIM as u32,
        TEST_NUM_V_HEADS as u32,
        TEST_NUM_K_HEADS as u32,
    )
    .expect("state advance");
    let wave = |len: usize, step: f32, amplitude: f32, bias: f32| {
        (0..len).map(|i| (i as f32 * step).sin() * amplitude + bias).collect::<Vec<f32>>()
    };
    let bf16_wave = |len: usize, step: f32, amplitude: f32, bias: f32| {
        wave(len, step, amplitude, bias).into_iter().map(bf16::from_f32).collect::<Vec<bf16>>()
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut sources = Vec::new();
    let mut bf16_scratch = |data: Vec<bf16>| scratch::<B, bf16>(&context, &mut command_buffer, &mut sources, &data);
    let path_k = bf16_scratch(bf16_wave(TREE_SIZE * TEST_NUM_K_HEADS * HEAD_DIM, 0.013, 0.08, 0.0));
    let path_v = bf16_scratch(bf16_wave(TREE_SIZE * TEST_NUM_V_HEADS * HEAD_DIM, 0.029, 0.4, 0.02));
    let q = bf16_scratch(bf16_wave(nodes * TEST_NUM_K_HEADS * HEAD_DIM, 0.017, 0.2, 0.01));
    let k = bf16_scratch(bf16_wave(nodes * TEST_NUM_K_HEADS * HEAD_DIM, 0.019, 0.18, -0.02));
    let v = bf16_scratch(bf16_wave(nodes * TEST_NUM_V_HEADS * HEAD_DIM, 0.023, 0.5, 0.03));
    let in_proj = bf16_scratch(bf16_wave(nodes * VERIFY_PROJ_DIM, 0.031, 1.5, -0.1));
    let mut f32_scratch = |data: Vec<f32>| scratch::<B, f32>(&context, &mut command_buffer, &mut sources, &data);
    let path = AcceptedPath {
        k: path_k,
        v: path_v,
        log_decay: f32_scratch(wave(TREE_SIZE * TEST_NUM_V_HEADS, 0.7, 0.05, -0.1)),
        beta: f32_scratch(wave(TREE_SIZE * TEST_NUM_V_HEADS, 0.011, 0.2, 0.4)),
        indices: indices.into(),
    };
    let log_decay = f32_scratch(wave(nodes * TEST_NUM_V_HEADS, 1.7, 0.2, -0.25));
    let beta = f32_scratch(wave(nodes * TEST_NUM_V_HEADS, 0.013, 0.2, 0.45));
    // Caterpillar: odd nodes are leaves, even nodes the spine, so every 16-node block branches.
    let trie: Vec<u32> =
        (0..nodes as u32).flat_map(|node| [node, [nodes as u32 - 1, node][node as usize % 2], 0]).collect();
    let trie = command_buffer.allocate_constant_from_slice(&trie).expect("trie");
    let norm_weight = create_buffer_with_data::<B, f32>(&context, &wave(HEAD_DIM, 0.05, 0.2, 1.0));
    let mut state =
        create_buffer_with_data::<B, f32>(&context, &wave(TEST_NUM_V_HEADS * HEAD_DIM * HEAD_DIM, 0.007, 0.05, -0.01));
    if !fused {
        path.encode_advance(&state_advance, &mut state, &mut command_buffer).expect("state advance");
    }
    let output = tree_verify
        .encode(
            TreeVerifyEncodeArguments {
                q: &q,
                k: &k,
                v: &v,
                trie: &trie,
                log_decay: &log_decay,
                beta: &beta,
                h0: &mut state,
                advance: fused.then_some((&path, &state_advance)),
                in_proj: &in_proj,
                norm_weight: &norm_weight,
                conv_dim: VERIFY_CONV_DIM as u32,
                total_proj_dim: VERIFY_PROJ_DIM as u32,
                norm_epsilon: 1e-6,
                tree_size: nodes as u32,
            },
            &mut command_buffer,
        )
        .expect("tree verify");
    let mut values = create_buffer::<B, bf16>(&context, nodes * TEST_NUM_V_HEADS * HEAD_DIM);
    command_buffer.encode_copy(&output, &mut values);
    drop(trie);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (buffer_to_vec(&*context, &values), buffer_to_vec(&*context, &state))
}

#[uzu_test]
fn test_tree_verify_advances_state() {
    for nodes in [16, 32] {
        for indices in [&PATH[..1], &PATH[..8]] {
            for_each_non_cpu_backend!(|B| {
                let (expected_values, expected_state) = advance_and_verify::<B>(indices, nodes, false);
                let (values, state) = advance_and_verify::<B>(indices, nodes, true);
                assert_eq_float(&expected_state, &state, 1e-6, "advanced state");
                assert_eq_float(&expected_values, &values, 2e-3, "tree values");
            });
        }
    }
}

#[cfg(backend = "metal")]
#[uzu_bench]
fn bench_state_advance(c: &mut Criterion) {
    const NUM_V_HEADS: usize = 48;
    const NUM_K_HEADS: usize = 16;
    const BENCHMARK: &str = "Metal/Kernel/GDNTreeVerify/StateAdvance";

    let context = crate::tests::util::shared_metal_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(
        &context,
        DataType::BF16,
        HEAD_DIM as u32,
        NUM_V_HEADS as u32,
        NUM_K_HEADS as u32,
    )
    .expect("kernel");
    let k_norm = create_buffer_with_data::<Metal, bf16>(
        &context,
        &vec![bf16::from_f32(0.001); TREE_SIZE * NUM_K_HEADS * HEAD_DIM],
    );
    let v = create_buffer_with_data::<Metal, bf16>(
        &context,
        &vec![bf16::from_f32(0.01); TREE_SIZE * NUM_V_HEADS * HEAD_DIM],
    );
    let log_decay = create_buffer_with_data::<Metal, f32>(&context, &vec![-0.01; TREE_SIZE * NUM_V_HEADS]);
    let beta = create_buffer_with_data::<Metal, f32>(&context, &vec![0.2; TREE_SIZE * NUM_V_HEADS]);
    let initial_state = vec![0.01; NUM_V_HEADS * HEAD_DIM * HEAD_DIM];
    let mut group = c.benchmark_group(BENCHMARK);
    group.sample_size(30).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));

    for accepted_len in [1usize, 4, 8, 16] {
        let accepted_indices = create_buffer_with_data::<Metal, u32>(&context, &PATH[..accepted_len]);
        let mut committed_states = ColdPool::new(buffer_size_bytes::<f32>(initial_state.len()), || {
            create_buffer_with_data::<Metal, f32>(&context, &initial_state)
        });
        group.bench_function(format!("L{accepted_len}"), |bencher| {
            iter_encode_loop_named::<Metal, _>(
                &context,
                bencher,
                &format!("{BENCHMARK}/L{accepted_len}"),
                |command_buffer| {
                    kernel.encode(
                        &k_norm,
                        &v,
                        &log_decay,
                        &beta,
                        &accepted_indices,
                        committed_states.next_mut(),
                        accepted_len as u32,
                        command_buffer,
                    );
                },
            );
        });
    }
}
