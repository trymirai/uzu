use half::bf16;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            kernel::{BuildTreeOutKernel, BuildTreeOutOutputTileMatmulDirectKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
};

#[derive(Clone, Copy)]
struct Shape {
    batch_size: usize,
    tree_size: usize,
    qk_heads: usize,
    value_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
}

struct Inputs {
    q: Vec<bf16>,
    prefix: Vec<f32>,
    qkd: Vec<f32>,
    u: Vec<bf16>,
    h0: Vec<bf16>,
    h0_indices: Vec<i32>,
}

#[derive(Clone, Copy)]
enum Candidate {
    Scalar,
    OutputTileMatmulDirectSimdgroup,
    OutputTileMatmulDirectMxu,
}

fn make_inputs(shape: Shape) -> Inputs {
    let q_len = shape.batch_size * shape.tree_size * shape.qk_heads * shape.head_k_dim;
    let prefix_len = shape.batch_size * shape.tree_size * shape.value_heads;
    let qkd_len = shape.batch_size * shape.value_heads * shape.tree_size * shape.tree_size;
    let u_len = shape.batch_size * shape.value_heads * shape.tree_size * shape.head_v_dim;
    let h0_pool = shape.batch_size + 1;
    let h0_len = h0_pool * shape.value_heads * shape.head_v_dim * shape.head_k_dim;

    Inputs {
        q: (0..q_len).map(|i| bf16::from_f32(((i as f32 * 0.017).sin() * 0.2) + 0.01)).collect(),
        prefix: (0..prefix_len)
            .map(|i| -((i % shape.tree_size) as f32) * 0.01 - ((i % shape.value_heads) as f32) * 0.003)
            .collect(),
        qkd: (0..qkd_len).map(|i| ((i as f32 * 0.013).cos() * 0.1) - 0.02).collect(),
        u: (0..u_len).map(|i| bf16::from_f32(((i as f32 * 0.011).sin() * 0.3) + 0.04)).collect(),
        h0: (0..h0_len).map(|i| bf16::from_f32(((i as f32 * 0.019).cos() * 0.2) - 0.01)).collect(),
        h0_indices: (0..shape.batch_size)
            .map(|i| {
                if i + 1 == shape.batch_size {
                    -1
                } else {
                    i as i32
                }
            })
            .collect(),
    }
}

fn run_cpu(
    shape: Shape,
    inputs: &Inputs,
    use_h0: bool,
) -> Vec<bf16> {
    let context = <Cpu as Backend>::Context::new().expect("Failed to create CPU context");
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::BuildTreeOutKernel::new(&context, DataType::BF16, use_h0)
        .expect("Failed to create CPU BuildTreeOutKernel");
    run_encoded::<Cpu, _>(context.as_ref(), shape, inputs, use_h0, |encoder, q, prefix, qkd, u, h0, h0_indices, o| {
        kernel.encode(
            q,
            prefix,
            qkd,
            u,
            h0,
            h0_indices,
            o,
            (shape.head_k_dim as f32).sqrt().recip(),
            shape.batch_size as u32,
            shape.tree_size as u32,
            shape.qk_heads as u32,
            shape.value_heads as u32,
            shape.head_k_dim as u32,
            shape.head_v_dim as u32,
            encoder,
        );
    })
}

fn run_candidate<B: Backend>(
    shape: Shape,
    inputs: &Inputs,
    use_h0: bool,
    candidate: Candidate,
) -> Vec<bf16> {
    let context = B::Context::new().expect("Failed to create Context");
    match candidate {
        Candidate::Scalar => {
            let kernel =
                <<B as Backend>::Kernels as Kernels>::BuildTreeOutKernel::new(&context, DataType::BF16, use_h0)
                    .expect("BuildTreeOutKernel");
            run_encoded::<B, _>(
                context.as_ref(),
                shape,
                inputs,
                use_h0,
                |encoder, q, prefix, qkd, u, h0, h0_indices, o| {
                    kernel.encode(
                        q,
                        prefix,
                        qkd,
                        u,
                        h0,
                        h0_indices,
                        o,
                        (shape.head_k_dim as f32).sqrt().recip(),
                        shape.batch_size as u32,
                        shape.tree_size as u32,
                        shape.qk_heads as u32,
                        shape.value_heads as u32,
                        shape.head_k_dim as u32,
                        shape.head_v_dim as u32,
                        encoder,
                    );
                },
            )
        },
        Candidate::OutputTileMatmulDirectSimdgroup | Candidate::OutputTileMatmulDirectMxu => {
            let use_mxu = matches!(candidate, Candidate::OutputTileMatmulDirectMxu);
            let kernel = <<B as Backend>::Kernels as Kernels>::BuildTreeOutOutputTileMatmulDirectKernel::new(
                &context,
                DataType::BF16,
                use_h0,
                use_mxu,
            )
            .expect("BuildTreeOutOutputTileMatmulDirectKernel");
            run_encoded::<B, _>(
                context.as_ref(),
                shape,
                inputs,
                use_h0,
                |encoder, q, prefix, qkd, u, h0, h0_indices, o| {
                    kernel.encode(
                        q,
                        prefix,
                        qkd,
                        u,
                        h0,
                        h0_indices,
                        o,
                        (shape.head_k_dim as f32).sqrt().recip(),
                        shape.batch_size as u32,
                        shape.tree_size as u32,
                        shape.qk_heads as u32,
                        shape.value_heads as u32,
                        shape.head_k_dim as u32,
                        shape.head_v_dim as u32,
                        encoder,
                    );
                },
            )
        },
    }
}

fn run_encoded<
    B: Backend,
    F: FnOnce(
        &mut Encoder<B>,
        &Allocation<B>,
        &Allocation<B>,
        &Allocation<B>,
        &Allocation<B>,
        Option<&Allocation<B>>,
        Option<&Allocation<B>>,
        &mut Allocation<B>,
    ),
>(
    context: &B::Context,
    shape: Shape,
    inputs: &Inputs,
    use_h0: bool,
    encode: F,
) -> Vec<bf16> {
    let q = context.create_array_from(&[inputs.q.len()], &inputs.q).into_allocation();
    let prefix = context.create_array_from(&[inputs.prefix.len()], &inputs.prefix).into_allocation();
    let qkd = context.create_array_from(&[inputs.qkd.len()], &inputs.qkd).into_allocation();
    let u = context.create_array_from(&[inputs.u.len()], &inputs.u).into_allocation();
    let h0 = use_h0.then(|| context.create_array_from(&[inputs.h0.len()], &inputs.h0).into_allocation());
    let h0_indices =
        use_h0.then(|| context.create_array_from(&[inputs.h0_indices.len()], &inputs.h0_indices).into_allocation());
    let mut o = context
        .create_array_uninitialized(
            &[shape.batch_size * shape.tree_size * shape.value_heads * shape.head_v_dim],
            DataType::BF16,
        )
        .into_allocation();

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");
    encode(&mut encoder, &q, &prefix, &qkd, &u, h0.as_ref(), h0_indices.as_ref(), &mut o);
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&o)
}

fn check_shape(shape: Shape) {
    let inputs = make_inputs(shape);
    let candidates =
        [Candidate::Scalar, Candidate::OutputTileMatmulDirectSimdgroup, Candidate::OutputTileMatmulDirectMxu];

    for use_h0 in [false, true] {
        let expected = run_cpu(shape, &inputs, use_h0);
        for_each_non_cpu_backend!(|B| {
            let context = <B as Backend>::Context::new().expect("Failed to create Context");
            for candidate in candidates {
                if matches!(candidate, Candidate::OutputTileMatmulDirectMxu) && !context.supports_mxu() {
                    continue;
                }
                let actual = run_candidate::<B>(shape, &inputs, use_h0, candidate);
                let msg = format!(
                    "backend {} candidate {} use_h0 {use_h0} B{}_T{}_QK{}_HV{}_K{}_V{}",
                    std::any::type_name::<B>(),
                    candidate_name(candidate),
                    shape.batch_size,
                    shape.tree_size,
                    shape.qk_heads,
                    shape.value_heads,
                    shape.head_k_dim,
                    shape.head_v_dim
                );
                assert_eq_float::<bf16>(&expected, &actual, 0.08, &msg);
            }
        });
    }
}

fn candidate_name(candidate: Candidate) -> &'static str {
    match candidate {
        Candidate::Scalar => "Scalar",
        Candidate::OutputTileMatmulDirectSimdgroup => "OutputTileMatmulDirectSimdgroup",
        Candidate::OutputTileMatmulDirectMxu => "OutputTileMatmulDirectMxu",
    }
}

#[uzu_test]
fn test_build_tree_out_candidates_match_cpu_small() {
    check_shape(Shape {
        batch_size: 2,
        tree_size: 17,
        qk_heads: 2,
        value_heads: 6,
        head_k_dim: 32,
        head_v_dim: 32,
    });
}

#[uzu_test]
fn test_build_tree_out_candidates_match_cpu_targetish() {
    check_shape(Shape {
        batch_size: 1,
        tree_size: 49,
        qk_heads: 16,
        value_heads: 48,
        head_k_dim: 128,
        head_v_dim: 128,
    });
}
