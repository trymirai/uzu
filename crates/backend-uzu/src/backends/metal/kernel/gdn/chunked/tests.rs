use half::bf16;
use num_traits::cast;
use proc_macros::uzu_test;

use super::*;
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Context, Kernels,
            kernel::{DeltaNetNormGateKernel, DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel},
        },
        metal::Metal,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

type BackendKernels<B> = <B as Backend>::Kernels;

const NUM_V_HEADS: usize = 4;
const NUM_K_HEADS: usize = 1;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;

#[derive(Clone, Copy)]
enum PrefillMode {
    Recurrent,
    Chunked,
}

struct TestCase {
    suffix_len: usize,
    in_proj: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_weight: Vec<f32>,
    key_dim: usize,
    value_dim: usize,
    total_proj_dim: usize,
}

fn test_case(suffix_len: usize) -> TestCase {
    let key_dim = NUM_K_HEADS * HEAD_K_DIM;
    let value_dim = NUM_V_HEADS * HEAD_V_DIM;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + NUM_V_HEADS + NUM_V_HEADS;

    TestCase {
        suffix_len,
        in_proj: (0..suffix_len * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect(),
        a_log: (0..NUM_V_HEADS).map(|i| -1.5 + (i as f32) * 0.05).collect(),
        dt_bias: (0..NUM_V_HEADS).map(|i| 0.3 + (i as f32) * 0.02).collect(),
        norm_weight: (0..HEAD_V_DIM).map(|i| 0.9 + (i as f32) * 0.001).collect(),
        key_dim,
        value_dim,
        total_proj_dim,
    }
}

fn run_prefill<T: ArrayElement>(
    context: &MetalContext,
    case: &TestCase,
    mode: PrefillMode,
) -> (Vec<f32>, Vec<f32>) {
    let in_proj_data: Vec<T> = case.in_proj.iter().copied().map(|value| cast(value).unwrap()).collect();
    let in_proj = alloc_allocation_with_data::<Metal, T>(&context, &in_proj_data);
    let a_log = alloc_allocation_with_data::<Metal, f32>(&context, &case.a_log);
    let dt_bias = alloc_allocation_with_data::<Metal, f32>(&context, &case.dt_bias);
    let norm_weight = alloc_allocation_with_data::<Metal, f32>(&context, &case.norm_weight);
    let state_len = NUM_V_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let mut state = alloc_allocation::<Metal, f32>(&context, state_len);
    let mut out = alloc_allocation::<Metal, T>(&context, case.suffix_len * case.value_dim);
    let mut q = alloc_allocation::<Metal, f32>(&context, case.suffix_len * case.key_dim);
    let mut k = alloc_allocation::<Metal, f32>(&context, case.suffix_len * case.key_dim);
    let mut beta = alloc_allocation::<Metal, f32>(&context, case.suffix_len * NUM_V_HEADS);
    let mut decay = alloc_allocation::<Metal, f32>(&context, case.suffix_len * NUM_V_HEADS);
    let mut encoder = Encoder::new(context).expect("encoder");

    match mode {
        PrefillMode::Recurrent => {
            <BackendKernels<Metal> as Kernels>::DeltaNetPrefillPrepKernel::new(
                &context,
                T::data_type(),
                HEAD_K_DIM as u32,
                false,
            )
            .expect("prep")
            .encode(
                &in_proj,
                &a_log,
                &dt_bias,
                &mut q,
                &mut k,
                &mut beta,
                &mut decay,
                NUM_V_HEADS as u32,
                NUM_K_HEADS as u32,
                case.key_dim as u32,
                case.value_dim as u32,
                case.suffix_len as u32,
                &mut encoder,
            );
            <BackendKernels<Metal> as Kernels>::DeltaNetPrefillKernel::new(&context, T::data_type(), HEAD_K_DIM as u32)
                .expect("prefill")
                .encode(
                    &q,
                    &k,
                    &beta,
                    &decay,
                    &in_proj,
                    &mut state,
                    &mut out,
                    NUM_V_HEADS as u32,
                    NUM_K_HEADS as u32,
                    HEAD_V_DIM as u32,
                    case.key_dim as u32,
                    case.value_dim as u32,
                    case.suffix_len as u32,
                    HEAD_V_DIM.div_ceil(16) as u32,
                    &mut encoder,
                );
        },
        PrefillMode::Chunked => {
            <BackendKernels<Metal> as Kernels>::DeltaNetChunkedPrefill::new(
                &context,
                T::data_type(),
                HEAD_K_DIM as u32,
            )
            .expect("chunked")
            .expect("chunked unsupported")
            .encode(
                DeltaNetChunkedPrefillArgs {
                    in_projected: &in_proj,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    ssm_state: &mut state,
                    delta_output: &mut out,
                    num_heads: NUM_V_HEADS as u32,
                    num_groups: NUM_K_HEADS as u32,
                    value_head_dim: HEAD_V_DIM as u32,
                    key_dim: case.key_dim as u32,
                    value_dim: case.value_dim as u32,
                    suffix_len: case.suffix_len,
                },
                &mut encoder,
            )
            .expect("chunked encode");
        },
    }

    <BackendKernels<Metal> as Kernels>::DeltaNetNormGateKernel::new(&context, T::data_type()).expect("norm").encode(
        &mut out,
        &in_proj,
        &norm_weight,
        NUM_V_HEADS as u32,
        HEAD_V_DIM as u32,
        case.value_dim as u32,
        (2 * case.key_dim + case.value_dim) as u32,
        case.total_proj_dim as u32,
        1e-6,
        case.suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (
        allocation_to_vec::<Metal, T>(&out).into_iter().map(|value| cast(value).unwrap()).collect(),
        allocation_to_vec(&state),
    )
}

#[uzu_test]
fn chunked_prefill_matches_recurrent_prefill() {
    let context = <Metal as Backend>::Context::new().expect("metal context");

    for suffix_len in [CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 1, CHUNK_SIZE * 2 + 1] {
        let case = test_case(suffix_len);
        let (recurrent_out, recurrent_state) = run_prefill::<bf16>(&context, &case, PrefillMode::Recurrent);
        let (chunked_out, chunked_state) = run_prefill::<bf16>(&context, &case, PrefillMode::Chunked);

        assert!(
            is_close::default().abs_tol(5e-3).rel_tol(1e-3).all_close(chunked_out.iter().copied(), recurrent_out),
            "chunked output differs at T={suffix_len}"
        );
        assert!(
            is_close::default().abs_tol(2e-4).rel_tol(1e-3).all_close(chunked_state.iter().copied(), recurrent_state),
            "chunked state differs at T={suffix_len}"
        );
    }
}
