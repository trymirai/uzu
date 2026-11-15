use futures::executor::block_on;
use half::{bf16, f16};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{
                JangtqCombineKernel, JangtqDownProjectTileKernel, JangtqDownReduceKernel, JangtqDownReduceCombineKernel,
                JangtqFusedGateUpKernel, JangtqFusedGateUpTileKernel, JangtqHadamard32Kernel, JangtqHadamard32F32Kernel,
                JangtqSortIndicesKernel, JangtqWeightedDownKernel, JangtqWeightedDownCombineKernel,
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
    },
};

// Codebooks from the MiniCPM-V-4.7 JANGH2 bundle (config.json jangtq.codebooks).
const ALPHA_2BIT: f32 = 0.893;
const BETA_2BIT: f32 = 0.05065;
const ALPHA_3BIT: f32 = 0.47125;
const BETA_3BIT: f32 = 0.0116;

fn run_hadamard32<B: Backend>(
    input: &[bf16],
    rows: u32,
    row_length: u32,
) -> Vec<f32> {
    let context = create_context::<B>();
    let x = create_buffer_with_data::<B, bf16>(&context, input);
    let mut out = create_buffer::<B, f32>(&context, input.len());
    let kernel = <B::Kernels as Kernels>::JangtqHadamard32Kernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&x, &mut out, row_length, rows, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_hadamard32_matches_cpu() {
    let rows = 3;
    let row_length = 256u32;
    let input: Vec<bf16> =
        (0..(rows * row_length) as usize).map(|i| bf16::from_f32((((i * 13 + 5) % 29) as f32 - 14.0) * 0.1)).collect();
    let expected = run_hadamard32::<Cpu>(&input, rows, row_length);
    // The butterfly is exact IEEE f32 arithmetic on both backends.
    for_each_non_cpu_backend!(|B| {
        let actual = run_hadamard32::<B>(&input, rows, row_length);
        assert_eq_float(&expected, &actual, 1e-6, "JangtqHadamard32");
    });
}

struct FusedGateUpCase {
    packed_g: Vec<u32>,
    scales_g: Vec<f16>,
    packed_u: Vec<u32>,
    scales_u: Vec<f16>,
    x: Vec<f32>,
    indices: Vec<u32>,
    tokens: u32,
}

fn run_fused_gate_up<B: Backend>(
    case: &FusedGateUpCase,
    k: u32,
    n: u32,
    _experts: u32,
) -> Vec<f32> {
    let context = create_context::<B>();
    let top_k = 8u32;
    let dispatches = case.tokens * top_k;
    let words = k * 2 / 32;
    let x = create_buffer_with_data::<B, f32>(&context, &case.x);
    let packed_g = create_buffer_with_data::<B, u32>(&context, &case.packed_g);
    let scales_g = create_buffer_with_data::<B, f16>(&context, &case.scales_g);
    let packed_u = create_buffer_with_data::<B, u32>(&context, &case.packed_u);
    let scales_u = create_buffer_with_data::<B, f16>(&context, &case.scales_u);
    let indices = create_buffer_with_data::<B, u32>(&context, &case.indices);
    let mut out = create_buffer::<B, f32>(&context, (dispatches * n) as usize);

    let kernel = <B::Kernels as Kernels>::JangtqFusedGateUpKernel::new(&context, 2, 2).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &x,
        &packed_g,
        &scales_g,
        &packed_u,
        &scales_u,
        &indices,
        ALPHA_2BIT,
        BETA_2BIT,
        ALPHA_2BIT,
        BETA_2BIT,
        k,
        n,
        words,
        words,
        top_k,
        dispatches,
        0.0,
        &mut out,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_fused_gate_up_matches_cpu() {
    // Production geometry: K = hidden 2048, N = 2 x moe_intermediate 512.
    let k = 2048u32;
    let n = 512u32;
    let experts = 8u32;
    let words = (k * 2 / 32) as usize;
    let packed: Vec<u32> = (0..(experts as usize * n as usize * words))
        .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
        .collect();
    let scales: Vec<f16> = (0..(experts as usize * n as usize))
        .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
        .collect();
    for tokens in [1u32, 3] {
        let top_k = 8u32;
        let dispatches = tokens * top_k;
        let x: Vec<f32> =
            (0..(tokens as usize * k as usize)).map(|i| (((i * 31 + 17) % 53) as f32 - 26.0) * 0.05).collect();
        let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 5 + 3) as u32) % experts).collect();
        let case = FusedGateUpCase {
            packed_g: packed.clone(),
            scales_g: scales.clone(),
            packed_u: packed.clone(),
            scales_u: scales.clone(),
            x,
            indices,
            tokens,
        };
        let expected = run_fused_gate_up::<Cpu>(&case, k, n, experts);
        for_each_non_cpu_backend!(|B| {
            // f32 outputs with reference-ordered accumulation; 1e-4 covers simd_sum tree
            // reordering (values reach ~1e2, so ~2 ulp at that magnitude is ~3e-5).
            assert_eq_float(&expected, &run_fused_gate_up::<B>(&case, k, n, experts), 1e-4, "JangtqFusedGateUp");
        });
    }
}

fn run_fused_gate_up_sorted<B: Backend>(
    case: &FusedGateUpCase,
    k: u32,
    n: u32,
    _experts: u32,
) -> Vec<f32> {
    let context = create_context::<B>();
    let top_k = 8u32;
    let dispatches = case.tokens * top_k;
    let words = k * 2 / 32;

    let x = create_buffer_with_data::<B, f32>(&context, &case.x);
    let packed_g = create_buffer_with_data::<B, u32>(&context, &case.packed_g);
    let scales_g = create_buffer_with_data::<B, f16>(&context, &case.scales_g);
    let packed_u = create_buffer_with_data::<B, u32>(&context, &case.packed_u);
    let scales_u = create_buffer_with_data::<B, f16>(&context, &case.scales_u);
    let indices = create_buffer_with_data::<B, u32>(&context, &case.indices);
    let mut sorted_indices = create_buffer::<B, u32>(&context, dispatches as usize);
    let mut expert_offsets = create_buffer::<B, u32>(&context, 256);
    let mut expert_counts = create_buffer::<B, u32>(&context, 256);
    let mut out = create_buffer::<B, f32>(&context, (dispatches * n) as usize);

    let sort_kernel = <B::Kernels as Kernels>::JangtqSortIndicesKernel::new(&context).unwrap();
    let gate_up_kernel = <B::Kernels as Kernels>::JangtqFusedGateUpKernel::new(&context, 2, 2).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    sort_kernel.encode(
        &indices,
        &mut sorted_indices,
        &mut expert_offsets,
        &mut expert_counts,
        dispatches,
        &mut command_buffer,
    );
    gate_up_kernel.encode(
        &x,
        &packed_g,
        &scales_g,
        &packed_u,
        &scales_u,
        &sorted_indices,
        ALPHA_2BIT,
        BETA_2BIT,
        ALPHA_2BIT,
        BETA_2BIT,
        k,
        n,
        words,
        words,
        top_k,
        dispatches,
        0.0,
        &mut out,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_fused_gate_up_sorted_matches_cpu() {
    let k = 2048u32;
    let n = 512u32;
    for &experts in &[8u32, 256u32] {
        let words = (k * 2 / 32) as usize;
        let packed: Vec<u32> = (0..(experts as usize * n as usize * words))
            .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
            .collect();
        let scales: Vec<f16> = (0..(experts as usize * n as usize))
            .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
            .collect();
        for tokens in [1u32, 3, 10] {
            let top_k = 8u32;
            let dispatches = tokens * top_k;
            let x: Vec<f32> =
                (0..(tokens as usize * k as usize)).map(|i| (((i * 31 + 17) % 53) as f32 - 26.0) * 0.05).collect();
            let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 17 + 3) as u32) % experts).collect();
            let case = FusedGateUpCase {
                packed_g: packed.clone(),
                scales_g: scales.clone(),
                packed_u: packed.clone(),
                scales_u: scales.clone(),
                x,
                indices,
                tokens,
            };
            let expected = run_fused_gate_up::<Cpu>(&case, k, n, experts);
            for_each_non_cpu_backend!(|B| {
                assert_eq_float(&expected, &run_fused_gate_up_sorted::<B>(&case, k, n, experts), 1e-4, "JangtqFusedGateUpSorted");
            });
        }
    }
}

fn run_fused_gate_up_tile<B: Backend>(
    case: &FusedGateUpCase,
    k: u32,
    n: u32,
    _experts: u32,
) -> Vec<f32> {
    let context = create_context::<B>();
    let top_k = 8u32;
    let dispatches = case.tokens * top_k;
    let words = k * 2 / 32;

    let x = create_buffer_with_data::<B, f32>(&context, &case.x);
    let packed_g = create_buffer_with_data::<B, u32>(&context, &case.packed_g);
    let scales_g = create_buffer_with_data::<B, f16>(&context, &case.scales_g);
    let packed_u = create_buffer_with_data::<B, u32>(&context, &case.packed_u);
    let scales_u = create_buffer_with_data::<B, f16>(&context, &case.scales_u);
    let indices = create_buffer_with_data::<B, u32>(&context, &case.indices);
    let mut sorted_indices = create_buffer::<B, u32>(&context, dispatches as usize);
    let mut expert_offsets = create_buffer::<B, u32>(&context, 256);
    let mut expert_counts = create_buffer::<B, u32>(&context, 256);
    let mut out = create_buffer::<B, f32>(&context, (dispatches * n) as usize);

    let sort_kernel = <B::Kernels as Kernels>::JangtqSortIndicesKernel::new(&context).unwrap();
    let gate_up_tile_kernel = <B::Kernels as Kernels>::JangtqFusedGateUpTileKernel::new(&context, 2, 2).unwrap();
    let hadamard_kernel = <B::Kernels as Kernels>::JangtqHadamard32F32Kernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    sort_kernel.encode(
        &indices,
        &mut sorted_indices,
        &mut expert_offsets,
        &mut expert_counts,
        dispatches,
        &mut command_buffer,
    );
    gate_up_tile_kernel.encode(
        &x,
        &packed_g,
        &scales_g,
        &packed_u,
        &scales_u,
        &sorted_indices,
        &expert_offsets,
        &expert_counts,
        ALPHA_2BIT,
        BETA_2BIT,
        ALPHA_2BIT,
        BETA_2BIT,
        k,
        n,
        words,
        words,
        top_k,
        dispatches,
        0.0,
        &mut out,
        &mut command_buffer,
    );
    hadamard_kernel.encode(&mut out, n, dispatches, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_fused_gate_up_tile_matches_cpu() {
    let k = 2048u32;
    let n = 512u32;
    for &experts in &[8u32, 256u32] {
        let words = (k * 2 / 32) as usize;
        let packed: Vec<u32> = (0..(experts as usize * n as usize * words))
            .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
            .collect();
        let scales: Vec<f16> = (0..(experts as usize * n as usize))
            .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
            .collect();
        for tokens in [1u32, 3, 10] {
            let top_k = 8u32;
            let dispatches = tokens * top_k;
            let x: Vec<f32> =
                (0..(tokens as usize * k as usize)).map(|i| (((i * 31 + 17) % 53) as f32 - 26.0) * 0.05).collect();
            let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 17 + 3) as u32) % experts).collect();
            let case = FusedGateUpCase {
                packed_g: packed.clone(),
                scales_g: scales.clone(),
                packed_u: packed.clone(),
                scales_u: scales.clone(),
                x,
                indices,
                tokens,
            };
            let expected = run_fused_gate_up::<Cpu>(&case, k, n, experts);
            for_each_non_cpu_backend!(|B| {
                assert_eq_float(&expected, &run_fused_gate_up_tile::<B>(&case, k, n, experts), 1e-4, "JangtqFusedGateUpTile");
            });
        }
    }
}

fn run_benchmark_for_backend<B: Backend>(
    k: u32,
    n: u32,
    _experts: u32,
    _tokens: u32,
    top_k: u32,
    dispatches: u32,
    words: u32,
    x: &[f32],
    packed: &[u32],
    scales: &[f16],
    indices: &[u32],
) {
    use std::time::Instant;
    let context = create_context::<B>();
    let x_buf = create_buffer_with_data::<B, f32>(&context, x);
    let packed_g = create_buffer_with_data::<B, u32>(&context, packed);
    let scales_g = create_buffer_with_data::<B, f16>(&context, scales);
    let packed_u = create_buffer_with_data::<B, u32>(&context, packed);
    let scales_u = create_buffer_with_data::<B, f16>(&context, scales);
    let indices_buf = create_buffer_with_data::<B, u32>(&context, indices);
    let mut sorted_indices = create_buffer::<B, u32>(&context, dispatches as usize);
    let mut expert_offsets = create_buffer::<B, u32>(&context, 256);
    let mut expert_counts = create_buffer::<B, u32>(&context, 256);
    let mut baseline_out = create_buffer::<B, f32>(&context, (dispatches * n) as usize);
    let mut tile_out = create_buffer::<B, f32>(&context, (dispatches * n) as usize);

    let sort_kernel = <B::Kernels as Kernels>::JangtqSortIndicesKernel::new(&context).unwrap();
    let baseline_kernel = <B::Kernels as Kernels>::JangtqFusedGateUpKernel::new(&context, 2, 2).unwrap();
    let tile_kernel = <B::Kernels as Kernels>::JangtqFusedGateUpTileKernel::new(&context, 2, 2).unwrap();
    let hadamard_kernel = <B::Kernels as Kernels>::JangtqHadamard32F32Kernel::new(&context).unwrap();

    // 1. Sort indices once
    let mut cmd_sort = context.create_command_buffer(None, None).unwrap();
    sort_kernel.encode(&indices_buf, &mut sorted_indices, &mut expert_offsets, &mut expert_counts, dispatches, &mut cmd_sort);
    block_on(cmd_sort.end_encoding().submit().wait_until_completed()).unwrap();

    // 2. Warmup & Parity Check
    {
        let mut cmd = context.create_command_buffer(None, None).unwrap();
        baseline_kernel.encode(&x_buf, &packed_g, &scales_g, &packed_u, &scales_u, &sorted_indices,
            ALPHA_2BIT, BETA_2BIT, ALPHA_2BIT, BETA_2BIT, k, n, words, words, top_k, dispatches, 0.0,
            &mut baseline_out, &mut cmd);
        block_on(cmd.end_encoding().submit().wait_until_completed()).unwrap();
    }
    {
        let mut cmd = context.create_command_buffer(None, None).unwrap();
        tile_kernel.encode(&x_buf, &packed_g, &scales_g, &packed_u, &scales_u, &sorted_indices,
            &expert_offsets, &expert_counts, ALPHA_2BIT, BETA_2BIT, ALPHA_2BIT, BETA_2BIT, k, n, words, words,
            top_k, dispatches, 0.0, &mut tile_out, &mut cmd);
        hadamard_kernel.encode(&mut tile_out, n, dispatches, &mut cmd);
        block_on(cmd.end_encoding().submit().wait_until_completed()).unwrap();
    }

    let baseline_vec = buffer_to_vec::<B, f32>(&context, &baseline_out);
    let tile_vec = buffer_to_vec::<B, f32>(&context, &tile_out);
    assert_eq_float(&baseline_vec, &tile_vec, 1e-4, "JangtqFusedGateUp Baseline vs Tile Parity");
    println!("✅ [Parity Verified] 100% numerical match between Baseline and Tile kernel (eps < 1e-4)!");

    // 3. Timed Benchmark (20 runs each)
    let iters = 20;
    let t_base_start = Instant::now();
    for _ in 0..iters {
        let mut cmd = context.create_command_buffer(None, None).unwrap();
        baseline_kernel.encode(&x_buf, &packed_g, &scales_g, &packed_u, &scales_u, &sorted_indices,
            ALPHA_2BIT, BETA_2BIT, ALPHA_2BIT, BETA_2BIT, k, n, words, words, top_k, dispatches, 0.0,
            &mut baseline_out, &mut cmd);
        block_on(cmd.end_encoding().submit().wait_until_completed()).unwrap();
    }
    let base_elapsed = t_base_start.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

    let t_tile_start = Instant::now();
    for _ in 0..iters {
        let mut cmd = context.create_command_buffer(None, None).unwrap();
        tile_kernel.encode(&x_buf, &packed_g, &scales_g, &packed_u, &scales_u, &sorted_indices,
            &expert_offsets, &expert_counts, ALPHA_2BIT, BETA_2BIT, ALPHA_2BIT, BETA_2BIT, k, n, words, words,
            top_k, dispatches, 0.0, &mut tile_out, &mut cmd);
        hadamard_kernel.encode(&mut tile_out, n, dispatches, &mut cmd);
        block_on(cmd.end_encoding().submit().wait_until_completed()).unwrap();
    }
    let tile_elapsed = t_tile_start.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

    let speedup = base_elapsed / tile_elapsed;
    let saving = (base_elapsed - tile_elapsed) * 40.0; // 40 layers in model

    println!("\n📊 [Standalone Operator Performance Comparison]");
    println!("   - Current Baseline (Paired Dispatch):   {base_elapsed:.3} ms / layer (40 layers: {:.3} s)", base_elapsed * 40.0 / 1000.0);
    println!("   - New Tile (Cooperative Shared Mem):     {tile_elapsed:.3} ms / layer (40 layers: {:.3} s)", tile_elapsed * 40.0 / 1000.0);
    println!("   - Operator Speedup Ratio:                {speedup:.2}x faster");
    println!("   - Estimated Full Prefill Net Saving:     {saving:.1} ms");
    println!("==========================================================================\n");
}

#[uzu_test]
fn jangtq_fused_gate_up_standalone_benchmark() {
    let k = 2048u32;
    let n = 512u32;
    let experts = 256u32;
    let tokens = 674u32;
    let top_k = 8u32;
    let dispatches = tokens * top_k;
    let words = k * 2 / 32;

    println!("\n==========================================================================");
    println!("  MoE JangtqFusedGateUp Standalone Operator Benchmark (Release Mode)");
    println!("  Tokens: {tokens}, TopK: {top_k}, Dispatches: {dispatches}, Experts: {experts}");
    println!("  Hidden Dim K: {k}, Intermediate Dim N: {n}");
    println!("==========================================================================");

    let words_u = words as usize;
    let packed: Vec<u32> = (0..(experts as usize * n as usize * words_u))
        .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
        .collect();
    let scales: Vec<f16> = (0..(experts as usize * n as usize))
        .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
        .collect();
    let x: Vec<f32> =
        (0..(tokens as usize * k as usize)).map(|i| (((i * 31 + 17) % 53) as f32 - 26.0) * 0.05).collect();
    let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 17 + 3) as u32) % experts).collect();

    for_each_non_cpu_backend!(|B| {
        run_benchmark_for_backend::<B>(k, n, experts, tokens, top_k, dispatches, words, &x, &packed, &scales, &indices);
    });
}

struct WeightedDownCase {
    hidden: Vec<f32>,
    packed: Vec<u32>,
    scales: Vec<f16>,
    indices: Vec<u32>,
    scores: Vec<f32>,
    tokens: u32,
}

fn run_weighted_down<B: Backend>(
    case: &WeightedDownCase,
    h: u32,
    n: u32,
    bits: u32,
    _experts: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let top_k = 8u32;
    let words = (h * bits / 32) as usize;
    let hidden = create_buffer_with_data::<B, f32>(&context, &case.hidden);
    let packed = create_buffer_with_data::<B, u32>(&context, &case.packed);
    let scales = create_buffer_with_data::<B, f16>(&context, &case.scales);
    let indices = create_buffer_with_data::<B, u32>(&context, &case.indices);
    let scores = create_buffer_with_data::<B, f32>(&context, &case.scores);
    let mut out = create_buffer::<B, bf16>(&context, (case.tokens * n) as usize);

    let kernel = <B::Kernels as Kernels>::JangtqWeightedDownKernel::new(&context, DataType::BF16, bits).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &hidden,
        &packed,
        &scales,
        &indices,
        &scores,
        &mut out,
        if bits == 2 {
            ALPHA_2BIT
        } else {
            ALPHA_3BIT
        },
        if bits == 2 {
            BETA_2BIT
        } else {
            BETA_3BIT
        },
        h,
        n,
        words as u32,
        top_k,
        case.tokens,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_weighted_down_matches_cpu() {
    // Production geometry: down reads the 512-wide rotated activation and writes hidden 2048.
    let h = 512u32;
    let n = 2048u32;
    let experts = 8u32;
    let top_k = 8u32;
    for bits in [2u32, 3] {
        let words = (h * bits / 32) as usize;
        let packed: Vec<u32> = (0..(experts as usize * n as usize * words))
            .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
            .collect();
        let scales: Vec<f16> = (0..(experts as usize * n as usize))
            .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
            .collect();
        for tokens in [1u32, 3] {
            let dispatches = tokens * top_k;
            let hidden: Vec<f32> =
                (0..(dispatches as usize * h as usize)).map(|i| (((i * 23 + 9) % 31) as f32 - 15.0) * 0.1).collect();
            let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 5 + 3) as u32) % experts).collect();
            let scores: Vec<f32> = (0..dispatches as usize).map(|i| 1.0 / (1.0 + (i % 8) as f32)).collect();
            let case = WeightedDownCase {
                hidden,
                packed: packed.clone(),
                scales: scales.clone(),
                indices,
                scores,
                tokens,
            };
            let expected = run_weighted_down::<Cpu>(&case, h, n, bits, experts);
            for_each_non_cpu_backend!(|B| {
                // bf16 outputs: 2e-2 is at least two bf16 steps at magnitude ~1.
                assert_eq_float(
                    &expected,
                    &run_weighted_down::<B>(&case, h, n, bits, experts),
                    2e-2,
                    &format!("JangtqWeightedDown ({bits} bit, {tokens} tokens)"),
                );
            });
        }
    }
}

fn run_down_project_sorted<B: Backend>(
    case: &WeightedDownCase,
    h: u32,
    n: u32,
    bits: u32,
    _experts: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let top_k = 8u32;
    let dispatches = case.tokens * top_k;
    let words = (h * bits / 32) as usize;
    let hidden = create_buffer_with_data::<B, f32>(&context, &case.hidden);
    let packed = create_buffer_with_data::<B, u32>(&context, &case.packed);
    let scales = create_buffer_with_data::<B, f16>(&context, &case.scales);
    let indices = create_buffer_with_data::<B, u32>(&context, &case.indices);
    let scores = create_buffer_with_data::<B, f32>(&context, &case.scores);
    let mut sorted_indices = create_buffer::<B, u32>(&context, dispatches as usize);
    let mut expert_offsets = create_buffer::<B, u32>(&context, 256);
    let mut expert_counts = create_buffer::<B, u32>(&context, 256);
    let mut routed_dispatches = create_buffer::<B, f32>(&context, (dispatches * n) as usize);
    let mut out = create_buffer::<B, bf16>(&context, (case.tokens * n) as usize);

    let sort_kernel = <B::Kernels as Kernels>::JangtqSortIndicesKernel::new(&context).unwrap();
    let project_tile_kernel = <B::Kernels as Kernels>::JangtqDownProjectTileKernel::new(&context, bits).unwrap();
    let reduce_kernel = <B::Kernels as Kernels>::JangtqDownReduceKernel::new(&context, DataType::BF16, 256).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    sort_kernel.encode(
        &indices,
        &mut sorted_indices,
        &mut expert_offsets,
        &mut expert_counts,
        dispatches,
        &mut command_buffer,
    );
    project_tile_kernel.encode(
        &hidden,
        &packed,
        &scales,
        &sorted_indices,
        &expert_offsets,
        &expert_counts,
        &scores,
        &mut routed_dispatches,
        if bits == 2 { ALPHA_2BIT } else { ALPHA_3BIT },
        if bits == 2 { BETA_2BIT } else { BETA_3BIT },
        h,
        n,
        words as u32,
        top_k,
        dispatches,
        &mut command_buffer,
    );
    reduce_kernel.encode(
        &routed_dispatches,
        &mut out,
        n,
        case.tokens,
        top_k,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_down_project_sorted_matches_cpu() {
    let h = 512u32;
    let n = 2048u32;
    let experts = 256u32;
    let top_k = 8u32;
    for bits in [2u32, 3] {
        let words = (h * bits / 32) as usize;
        let packed: Vec<u32> = (0..(experts as usize * n as usize * words))
            .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
            .collect();
        let scales: Vec<f16> = (0..(experts as usize * n as usize))
            .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
            .collect();
        for tokens in [1u32, 3, 6] {
            let dispatches = tokens * top_k;
            let hidden: Vec<f32> =
                (0..(dispatches as usize * h as usize)).map(|i| (((i * 23 + 9) % 31) as f32 - 15.0) * 0.1).collect();
            let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 19 + 7) as u32) % experts).collect();
            let scores: Vec<f32> = (0..dispatches as usize).map(|i| 1.0 / (1.0 + (i % 8) as f32)).collect();
            let case = WeightedDownCase {
                hidden,
                packed: packed.clone(),
                scales: scales.clone(),
                indices,
                scores,
                tokens,
            };
            let expected = run_weighted_down::<Cpu>(&case, h, n, bits, experts);
            for_each_non_cpu_backend!(|B| {
                assert_eq_float(
                    &expected,
                    &run_down_project_sorted::<B>(&case, h, n, bits, experts),
                    5e-2,
                    &format!("JangtqDownProjectSorted ({bits} bit, {tokens} tokens)"),
                );
            });
        }
    }
}

fn run_jangtq_combine<B: Backend>(
    routed: &[bf16],
    shared_output: &[bf16],
    shared_gate: &[bf16],
    batch_dim: u32,
    model_dim: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let routed_buf = create_buffer_with_data::<B, bf16>(&context, routed);
    let shared_out_buf = create_buffer_with_data::<B, bf16>(&context, shared_output);
    let shared_gate_buf = create_buffer_with_data::<B, bf16>(&context, shared_gate);
    let mut output_buf = create_buffer::<B, bf16>(&context, (batch_dim * model_dim) as usize);

    let kernel = <B::Kernels as Kernels>::JangtqCombineKernel::new(&context, DataType::BF16, 256).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &routed_buf,
        &shared_out_buf,
        &shared_gate_buf,
        &mut output_buf,
        model_dim,
        batch_dim,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &output_buf)
}

#[uzu_test]
fn jangtq_combine_matches_cpu() {
    let model_dim = 2048u32;
    for batch_dim in [1u32, 3] {
        let routed: Vec<bf16> = (0..(batch_dim * model_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 3) % 71) as f32 - 35.0) * 0.05))
            .collect();
        let shared_output: Vec<bf16> = (0..(batch_dim * model_dim) as usize)
            .map(|i| bf16::from_f32((((i * 17 + 11) % 43) as f32 - 21.0) * 0.05))
            .collect();
        let shared_gate: Vec<bf16> = (0..batch_dim as usize).map(|i| bf16::from_f32((i as f32 - 1.5) * 0.7)).collect();
        let expected = run_jangtq_combine::<Cpu>(&routed, &shared_output, &shared_gate, batch_dim, model_dim);
        for_each_non_cpu_backend!(|B| {
            // bf16-intermediate rounding on both sides; fast::exp vs libm differs below
            // the bf16 quantum, so 2e-2 leaves room for one bf16 step.
            assert_eq_float(
                &expected,
                &run_jangtq_combine::<B>(&routed, &shared_output, &shared_gate, batch_dim, model_dim),
                2e-2,
                &format!("JangtqCombine ({batch_dim} tokens)"),
            );
        });
    }
}

fn run_jangtq_down_reduce_combine<B: Backend>(
    routed_dispatches: &[f32],
    shared_output: &[bf16],
    shared_gate: &[bf16],
    batch_dim: u32,
    model_dim: u32,
    routes_per_token: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let dispatches_buf = create_buffer_with_data::<B, f32>(&context, routed_dispatches);
    let shared_out_buf = create_buffer_with_data::<B, bf16>(&context, shared_output);
    let shared_gate_buf = create_buffer_with_data::<B, bf16>(&context, shared_gate);
    let mut output_buf = create_buffer::<B, bf16>(&context, (batch_dim * model_dim) as usize);

    let kernel = <B::Kernels as Kernels>::JangtqDownReduceCombineKernel::new(&context, DataType::BF16, 256).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &dispatches_buf,
        &shared_out_buf,
        &shared_gate_buf,
        &mut output_buf,
        model_dim,
        batch_dim,
        routes_per_token,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &output_buf)
}

#[uzu_test]
fn jangtq_down_reduce_combine_matches_cpu() {
    let model_dim = 2048u32;
    let routes_per_token = 8u32;
    for batch_dim in [1u32, 3] {
        let dispatches = (batch_dim * routes_per_token) as usize;
        let routed_dispatches: Vec<f32> = (0..(dispatches * model_dim as usize))
            .map(|i| (((i * 29 + 3) % 71) as f32 - 35.0) * 0.05)
            .collect();
        let shared_output: Vec<bf16> = (0..(batch_dim * model_dim) as usize)
            .map(|i| bf16::from_f32((((i * 17 + 11) % 43) as f32 - 21.0) * 0.05))
            .collect();
        let shared_gate: Vec<bf16> = (0..batch_dim as usize).map(|i| bf16::from_f32((i as f32 - 1.5) * 0.7)).collect();
        let expected = run_jangtq_down_reduce_combine::<Cpu>(
            &routed_dispatches,
            &shared_output,
            &shared_gate,
            batch_dim,
            model_dim,
            routes_per_token,
        );
        for_each_non_cpu_backend!(|B| {
            assert_eq_float(
                &expected,
                &run_jangtq_down_reduce_combine::<B>(
                    &routed_dispatches,
                    &shared_output,
                    &shared_gate,
                    batch_dim,
                    model_dim,
                    routes_per_token,
                ),
                5e-2,
                &format!("JangtqDownReduceCombine ({batch_dim} tokens)"),
            );
        });
    }
}

fn run_weighted_down_combine<B: Backend>(
    case: &WeightedDownCase,
    shared_output: &[bf16],
    shared_gate: &[bf16],
    h: u32,
    n: u32,
    bits: u32,
    _experts: u32,
) -> Vec<bf16> {
    let top_k = case.indices.len() as u32 / case.tokens;
    let words = (h * bits / 32) as usize;
    let context = create_context::<B>();
    let hidden = create_buffer_with_data::<B, _>(&context, &case.hidden);
    let packed = create_buffer_with_data::<B, _>(&context, &case.packed);
    let scales = create_buffer_with_data::<B, _>(&context, &case.scales);
    let indices = create_buffer_with_data::<B, _>(&context, &case.indices);
    let scores = create_buffer_with_data::<B, _>(&context, &case.scores);
    let shared_out_buf = create_buffer_with_data::<B, _>(&context, shared_output);
    let shared_gate_buf = create_buffer_with_data::<B, _>(&context, shared_gate);
    let mut out = create_buffer::<B, bf16>(&context, (case.tokens * n) as usize);

    let kernel = <B::Kernels as Kernels>::JangtqWeightedDownCombineKernel::new(&context, DataType::BF16, bits).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &hidden,
        &packed,
        &scales,
        &indices,
        &scores,
        &shared_out_buf,
        &shared_gate_buf,
        &mut out,
        if bits == 2 {
            ALPHA_2BIT
        } else {
            ALPHA_3BIT
        },
        if bits == 2 {
            BETA_2BIT
        } else {
            BETA_3BIT
        },
        h,
        n,
        words as u32,
        top_k,
        case.tokens,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &out)
}

#[uzu_test]
fn jangtq_weighted_down_combine_matches_cpu_and_baseline() {
    let h = 512u32;
    let n = 2048u32;
    let experts = 8u32;
    let top_k = 8u32;
    for bits in [2u32, 3] {
        let words = (h * bits / 32) as usize;
        let packed: Vec<u32> = (0..(experts as usize * n as usize * words))
            .map(|i| ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as u32)
            .collect();
        let scales: Vec<f16> = (0..(experts as usize * n as usize))
            .map(|i| f16::from_f32((((i * 11 + 7) % 13) as f32) * 0.01 + 0.02))
            .collect();
        for tokens in [1u32, 2] {
            let dispatches = tokens * top_k;
            let hidden: Vec<f32> =
                (0..(dispatches as usize * h as usize)).map(|i| (((i * 23 + 9) % 31) as f32 - 15.0) * 0.1).collect();
            let indices: Vec<u32> = (0..dispatches as usize).map(|i| ((i * 5 + 3) as u32) % experts).collect();
            let scores: Vec<f32> = (0..dispatches as usize).map(|i| 1.0 / (1.0 + (i % 8) as f32)).collect();
            let shared_output: Vec<bf16> = (0..(tokens * n) as usize)
                .map(|i| bf16::from_f32((((i * 19 + 5) % 37) as f32 - 18.0) * 0.05))
                .collect();
            let shared_gate: Vec<bf16> = (0..tokens as usize).map(|i| bf16::from_f32((i as f32 - 0.8) * 0.9)).collect();
            let case = WeightedDownCase {
                hidden,
                packed: packed.clone(),
                scales: scales.clone(),
                indices,
                scores,
                tokens,
            };

            // CPU reference for fused kernel:
            let expected_fused = run_weighted_down_combine::<Cpu>(&case, &shared_output, &shared_gate, h, n, bits, experts);

            // Separate CPU pipeline (run_weighted_down + run_jangtq_combine):
            let routed_cpu = run_weighted_down::<Cpu>(&case, h, n, bits, experts);
            let expected_separate = run_jangtq_combine::<Cpu>(&routed_cpu, &shared_output, &shared_gate, tokens, n);

            // Verify CPU fused matches CPU separate 100% exactly:
            assert_eq_float(&expected_fused, &expected_separate, 1e-6, "CPU Fused vs CPU Separate");

            // Verify non-CPU backend (Metal) matches CPU:
            for_each_non_cpu_backend!(|B| {
                let actual_fused = run_weighted_down_combine::<B>(&case, &shared_output, &shared_gate, h, n, bits, experts);
                assert_eq_float(
                    &expected_fused,
                    &actual_fused,
                    5e-2,
                    &format!("JangtqWeightedDownCombine ({bits} bit, {tokens} tokens)"),
                );
            });
        }
    }
}
