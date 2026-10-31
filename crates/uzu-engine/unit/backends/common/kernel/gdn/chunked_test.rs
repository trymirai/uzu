use futures::executor::block_on;
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::{
            DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel,
            delta_net_chunked_prefill::{CHUNK_SIZE, DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};
#[cfg(backend = "metal")]
use crate::{backends::metal::Metal, utils::downcast::downcast_ref};

type BackendKernels<B> = <B as Backend>::Kernels;
type ChunkedPrefill<B> = <BackendKernels<B> as Kernels>::DeltaNetChunkedPrefill;

const NUM_K_HEADS: usize = 2;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const KEY_DIM: usize = NUM_K_HEADS * HEAD_K_DIM;
const WARMUP_TOKENS: usize = 1024;

/// The in_proj columns of `num_v_heads` value heads: q, k and v, then z, then b and a.
fn total_proj_dim(num_v_heads: usize) -> usize {
    2 * KEY_DIM + 2 * num_v_heads * HEAD_V_DIM + 2 * num_v_heads
}

fn random_in_proj(
    tokens: usize,
    num_v_heads: usize,
) -> Vec<bf16> {
    let mut rng = SmallRng::seed_from_u64(7);
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let total_proj_dim = total_proj_dim(num_v_heads);
    (0..tokens * total_proj_dim)
        .map(|i| {
            let x: f32 = rng.sample(normal);
            let column = i % total_proj_dim;
            bf16::from_f32(if column < 2 * KEY_DIM + num_v_heads * HEAD_V_DIM {
                x / (1.0 + (-x).exp())
            } else if column >= total_proj_dim - num_v_heads {
                0.5 * x
            } else {
                x
            })
        })
        .collect()
}

/// `None` runs the recurrent kernels. The output is NaN-filled first so unwritten elements fail the comparison.
fn run_prefill<B: Backend>(
    context: &B::Context,
    num_v_heads: usize,
    in_proj_data: &[bf16],
    chunked: Option<&ChunkedPrefill<B>>,
    state: &mut B::GlobalGpuBuffer,
) -> Vec<f32> {
    let suffix_len = in_proj_data.len() / total_proj_dim(num_v_heads);
    let value_dim = num_v_heads * HEAD_V_DIM;
    // a_log = ln U(1, 16) and dt_bias = softplus^-1(logU(1e-3, 1e-1)) as in Qwen3-Next
    let spread = |i: usize| (i as f32 + 0.5) / num_v_heads as f32;
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| (1.0 + 15.0 * spread(i)).ln()).collect();
    let dt_bias: Vec<f32> =
        (0..num_v_heads).map(|i| (1e-3f32.ln() + 100f32.ln() * spread(i)).exp().exp_m1().ln()).collect();
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let in_proj_source = create_buffer_with_data::<B, bf16>(context, in_proj_data);
    let mut in_proj = command_buffer.allocate_scratch(size_of_val(in_proj_data)).unwrap();
    command_buffer.encode_copy(&in_proj_source, &mut in_proj);
    let a_log = create_buffer_with_data::<B, f32>(context, &a_log);
    let dt_bias = create_buffer_with_data::<B, f32>(context, &dt_bias);
    let mut out = command_buffer.allocate_scratch(suffix_len * value_dim * size_of::<bf16>()).unwrap();
    command_buffer.encode_fill(&mut out, 0xFF);

    if let Some(chunked) = chunked {
        chunked
            .encode(
                DeltaNetChunkedPrefillArgs {
                    in_projected: &in_proj,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    ssm_state: state,
                    delta_output: &mut out,
                    num_heads: num_v_heads as u32,
                    num_groups: NUM_K_HEADS as u32,
                    value_head_dim: HEAD_V_DIM as u32,
                    key_dim: KEY_DIM as u32,
                    value_dim: value_dim as u32,
                    suffix_len: suffix_len as u32,
                },
                &mut command_buffer,
            )
            .expect("chunked encode");
    } else {
        let mut q = create_buffer::<B, f32>(context, suffix_len * KEY_DIM);
        let mut k = create_buffer::<B, f32>(context, suffix_len * KEY_DIM);
        let mut beta = create_buffer::<B, f32>(context, suffix_len * num_v_heads);
        let mut decay = create_buffer::<B, f32>(context, suffix_len * num_v_heads);
        <BackendKernels<B> as Kernels>::DeltaNetPrefillPrepKernel::new(
            context,
            DataType::BF16,
            DataType::F32,
            HEAD_K_DIM as u32,
            false,
            false,
        )
        .expect("prep")
        .encode(
            &in_proj,
            &a_log,
            &dt_bias,
            &mut q,
            &mut k,
            None::<&mut B::GlobalGpuBuffer>,
            &mut beta,
            &mut decay,
            num_v_heads as u32,
            NUM_K_HEADS as u32,
            KEY_DIM as u32,
            value_dim as u32,
            suffix_len as u32,
            &mut command_buffer,
        );
        <BackendKernels<B> as Kernels>::DeltaNetPrefillKernel::new(context, DataType::BF16, HEAD_K_DIM as u32)
            .expect("prefill")
            .encode(
                &q,
                &k,
                &beta,
                &decay,
                &in_proj,
                state,
                &mut out,
                num_v_heads as u32,
                NUM_K_HEADS as u32,
                HEAD_V_DIM as u32,
                KEY_DIM as u32,
                value_dim as u32,
                suffix_len as u32,
                HEAD_V_DIM.div_ceil(16) as u32,
                &mut command_buffer,
            );
    }
    let mut output = create_buffer::<B, bf16>(context, suffix_len * value_dim);
    command_buffer.encode_copy(&out, &mut output);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, bf16>(context, &output).into_iter().map(f32::from).collect()
}

fn check<B: Backend>(
    context: &B::Context,
    num_v_heads: usize,
    chunked: &ChunkedPrefill<B>,
    state0: &[f32],
    in_proj: &[bf16],
    calls: usize,
    label: &str,
) {
    let mut chunked_state = create_buffer_with_data::<B, f32>(context, state0);
    let mut recurrent_state = create_buffer_with_data::<B, f32>(context, state0);
    let chunked_out: Vec<f32> = in_proj
        .chunks(in_proj.len() / calls)
        .flat_map(|tokens| run_prefill::<B>(context, num_v_heads, tokens, Some(chunked), &mut chunked_state))
        .collect();
    let recurrent_out = run_prefill::<B>(context, num_v_heads, in_proj, None, &mut recurrent_state);
    let (chunked_state, recurrent_state) =
        (buffer_to_vec::<B, f32>(context, &chunked_state), buffer_to_vec::<B, f32>(context, &recurrent_state));

    assert!(
        is_close::default().abs_tol(5e-3).rel_tol(1e-3).all_close(chunked_out, recurrent_out),
        "chunked output differs from recurrent ({label})"
    );
    assert!(
        is_close::default().abs_tol(1e-2).rel_tol(1e-3).all_close(chunked_state, recurrent_state),
        "chunked state differs from recurrent ({label})"
    );
}

fn chunked_prefill_matches_recurrent_prefill_impl<B: Backend>(v_per_k: usize) {
    let context = create_context::<B>();
    // Metal builds chunked prefill on MXU only.
    #[cfg(backend = "metal")]
    if downcast_ref::<<Metal as Backend>::Context>(&*context).is_some_and(|context| !context.supports_mxu) {
        return;
    }
    let num_v_heads = v_per_k * NUM_K_HEADS;
    let total_proj_dim = total_proj_dim(num_v_heads);
    let chunked = ChunkedPrefill::<B>::new(&context, DataType::BF16, HEAD_K_DIM as u32, v_per_k as u32)
        .unwrap()
        .expect("chunked prefill");
    let in_proj = random_in_proj(WARMUP_TOKENS + 4096, num_v_heads);
    let (warmup, tokens) = in_proj.split_at(WARMUP_TOKENS * total_proj_dim);
    let mut state = create_buffer_with_data::<B, f32>(&context, &vec![0.0; num_v_heads * HEAD_V_DIM * HEAD_K_DIM]);
    run_prefill::<B>(&context, num_v_heads, warmup, None, &mut state);
    let state0 = buffer_to_vec::<B, f32>(&context, &state);
    assert!(state0.iter().any(|value| *value != 0.0), "initial state is zero");

    for suffix_len in [2 * CHUNK_SIZE as usize + 1, 512, 1024] {
        let label = format!("T={suffix_len}");
        check::<B>(&context, num_v_heads, &chunked, &state0, &tokens[..suffix_len * total_proj_dim], 1, &label);
    }
    check::<B>(&context, num_v_heads, &chunked, &state0, tokens, 4, "carried 4x1024");
}

/// Value heads per key head: 1 as in Qwen3.5 0.8B, up to the 4 the chunked kernels share a workgroup among.
#[rstest]
#[test_attr(uzu_test)]
fn chunked_prefill_matches_recurrent_prefill(#[values(1, 3, 4)] v_per_k: usize) {
    for_each_non_cpu_backend!(|B| chunked_prefill_matches_recurrent_prefill_impl::<B>(v_per_k));
}
