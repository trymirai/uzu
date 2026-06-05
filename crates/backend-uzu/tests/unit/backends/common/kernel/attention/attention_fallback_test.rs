use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            kernel::{
                AttentionFallbackScatterScoresKernel, AttentionFallbackScatterValuesKernel, AttentionSinglePassKernel,
                SoftmaxKernel,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
    common::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
    data_type::DataType,
};

const NUM_HEADS: u32 = 4;
const NUM_GROUPS: u32 = 2;
const SEQ: u32 = 16;
const SUFFIX: u32 = 3;
const HEAD_DIM: u32 = 512;
const GQA: u32 = NUM_HEADS / NUM_GROUPS;

fn make_inputs() -> (Box<[bf16]>, Box<[bf16]>, Box<[bf16]>) {
    let q = (0..(NUM_HEADS * SUFFIX * HEAD_DIM) as usize)
        .map(|i| bf16::from_f32((i as f32 * 0.013 + 0.5).sin() * 0.5))
        .collect();
    let k = (0..(NUM_GROUPS * SEQ * HEAD_DIM) as usize)
        .map(|i| bf16::from_f32((i as f32 * 0.007 + 1.0).cos() * 0.5))
        .collect();
    let v = (0..(NUM_GROUPS * SEQ * HEAD_DIM) as usize)
        .map(|i| bf16::from_f32((i as f32 * 0.011 + 2.0).sin() * 0.5))
        .collect();
    (q, k, v)
}

// Ground truth: fused single-pass attention on the CPU backend.
fn reference(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    scale: f32,
) -> Vec<bf16> {
    let ctx = <Cpu as Backend>::Context::new().unwrap();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &ctx,
        DataType::BF16,
        HEAD_DIM,
        false,
        false,
        true,
        false,
        false,
    )
    .unwrap();
    let qa = alloc_allocation_with_data::<Cpu, bf16>(&ctx, q);
    let ka = alloc_allocation_with_data::<Cpu, bf16>(&ctx, k);
    let va = alloc_allocation_with_data::<Cpu, bf16>(&ctx, v);
    let mut out = alloc_allocation::<Cpu, bf16>(&ctx, (NUM_HEADS * SUFFIX * HEAD_DIM) as usize);
    let mut enc = Encoder::new(ctx.as_ref()).unwrap();
    kernel.encode(
        &qa,
        &ka,
        &va,
        &mut out,
        GQA,
        SEQ,
        SEQ * HEAD_DIM,
        HEAD_DIM,
        SEQ * HEAD_DIM,
        HEAD_DIM,
        None,
        scale,
        None::<&Allocation<Cpu>>,
        None,
        None::<&Allocation<Cpu>>,
        NUM_HEADS,
        SUFFIX,
        &mut enc,
    );
    enc.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, bf16>(&out)
}

// Production fallback pipeline (matmul → scatter → softmax → matmul → scatter)
// driven directly via the kernel-cache traits, for backends other than CPU.
fn pipeline_output<B: Backend>(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    scale: f32,
) -> Vec<bf16> {
    let ctx = B::Context::new().unwrap();
    let softmax = <<B as Backend>::Kernels as Kernels>::SoftmaxKernel::new(&ctx, DataType::BF16, false).unwrap();
    let scatter_scores = <<B as Backend>::Kernels as Kernels>::AttentionFallbackScatterScoresKernel::new(
        &ctx,
        DataType::BF16,
        false,
        true,
        false,
        false,
    )
    .unwrap();
    let scatter_values =
        <<B as Backend>::Kernels as Kernels>::AttentionFallbackScatterValuesKernel::new(&ctx, DataType::BF16).unwrap();
    let mut matmul =
        <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(&ctx, DataType::BF16, DataType::BF16, DataType::BF16)
            .unwrap();

    let qa = alloc_allocation_with_data::<B, bf16>(&ctx, q);
    let ka = alloc_allocation_with_data::<B, bf16>(&ctx, k);
    let va = alloc_allocation_with_data::<B, bf16>(&ctx, v);
    let mut scores = alloc_allocation::<B, bf16>(&ctx, (NUM_HEADS * SUFFIX * SEQ) as usize);
    let mut grp_s = alloc_allocation::<B, bf16>(&ctx, (GQA * SUFFIX * SEQ) as usize);
    let mut grp_o = alloc_allocation::<B, bf16>(&ctx, (GQA * SUFFIX * HEAD_DIM) as usize);
    let mut out = alloc_allocation::<B, bf16>(&ctx, (NUM_HEADS * SUFFIX * HEAD_DIM) as usize);

    let dt = DataType::BF16.size_in_bytes();
    let q_row = HEAD_DIM as usize * dt;
    let s_row = SEQ as usize * dt;
    let head_stride = (SEQ * HEAD_DIM) as usize;
    let mut enc = Encoder::new(ctx.as_ref()).unwrap();

    for g in 0..NUM_GROUPS {
        matmul
            .encode(
                MatmulArguments {
                    a: &qa,
                    a_offset: g as usize * GQA as usize * SUFFIX as usize * q_row,
                    b: MatmulB::FullPrecision {
                        b: &ka,
                    },
                    b_offset: g as usize * head_stride * dt,
                    b_leading_dimension: Some(HEAD_DIM),
                    b_transpose: true,
                    d: &mut grp_s,
                    d_transform: MatmulDOps {
                        ab_scale: scale,
                        accumulate: false,
                        bias: None,
                        rht_factors: None,
                    },
                    m: GQA * SUFFIX,
                    n: SEQ,
                    k: HEAD_DIM,
                },
                &mut enc,
            )
            .expect("encode failed");
        scatter_scores.encode(
            &grp_s,
            &mut scores,
            None,
            None::<&Allocation<B>>,
            None,
            g,
            GQA,
            SEQ,
            SUFFIX,
            GQA * SUFFIX * SEQ,
            &mut enc,
        );
    }
    softmax.encode(&mut scores, None::<&Allocation<B>>, SEQ, NUM_HEADS, SUFFIX, &mut enc);
    for g in 0..NUM_GROUPS {
        matmul
            .encode(
                MatmulArguments {
                    a: &scores,
                    a_offset: g as usize * GQA as usize * SUFFIX as usize * s_row,
                    b: MatmulB::FullPrecision {
                        b: &va,
                    },
                    b_offset: g as usize * head_stride * dt,
                    b_leading_dimension: Some(HEAD_DIM),
                    b_transpose: false,
                    d: &mut grp_o,
                    d_transform: MatmulDOps::none(),
                    m: GQA * SUFFIX,
                    n: HEAD_DIM,
                    k: SEQ,
                },
                &mut enc,
            )
            .expect("encode failed");
        scatter_values.encode(&grp_o, &mut out, g, GQA, SUFFIX, NUM_HEADS, HEAD_DIM, GQA * SUFFIX * HEAD_DIM, &mut enc);
    }
    enc.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, bf16>(&out)
}

#[uzu_test]
fn test_head_dim_512_bf16() {
    let (q, k, v) = make_inputs();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let expected = reference(&q, &k, &v, scale);
    for_each_non_cpu_backend!(|B| {
        let output = pipeline_output::<B>(&q, &k, &v, scale);
        assert_eq_float::<bf16>(&expected, &output, 1e-2, "AttentionFallback bf16");
    });
}
