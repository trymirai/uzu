use std::{
    fmt::Debug,
};

use half::bf16;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Encoder, Kernels, kernel::MoeRouterTopKKernel},
        cpu::Cpu,
    },
};

use crate::{common::helpers::create_context, uzu_test};

fn get_output<B: Backend, T: ArrayElement + Float>(
    input: &[T],
    weights: &[T],
    bias: &[T],
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
    renorm: bool,
) -> (Vec<i32>, Vec<T>) {
    let ctx = create_context::<B>();

    let input_array = ctx.create_array_from(&[input.len()], input, "");
    let weights_array = ctx.create_array_from(&[weights.len()], weights, "");
    let bias_array = ctx.create_array_from(&[bias.len()], bias, "");
    let mut ids = ctx.create_array_uninitialized(&[t * k], DataType::I32, "").into_allocation();
    let mut probs = ctx.create_array_uninitialized(&[t * k], T::data_type(), "").into_allocation();

    let kernel = <<B as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, T::data_type()).expect("kernel");
    let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_array.allocation(),
        weights_array.allocation(),
        bias_array.allocation(),
        &mut ids,
        &mut probs,
        t as u32,
        d_model as u32,
        e as u32,
        k as u32,
        renorm,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (
        crate::common::helpers::allocation_to_vec(&ids),
        crate::common::helpers::allocation_to_vec(&probs),
    )
}

fn run_router_topk_once<B: Backend, T: ArrayElement + Debug + Float>(
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
    renorm: bool,
) {
    let mut rng = StdRng::seed_from_u64(1234);
    let input: Vec<T> = (0..t * d_model).map(|_| T::from(rng.random_range(-1.0..1.0)).unwrap()).collect();
    let weight: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-1.0..1.0)).unwrap()).collect();
    let bias: Vec<T> = (0..e).map(|_| T::from(rng.random_range(-0.5..0.5)).unwrap()).collect();

    let (ids_ref, probs_ref) =
        get_output::<Cpu, T>(input.as_slice(), weight.as_slice(), bias.as_slice(), t, d_model, e, k, renorm);
    let (ids_gpu, probs_gpu) =
        get_output::<B, T>(input.as_slice(), weight.as_slice(), bias.as_slice(), t, d_model, e, k, renorm);
    assert_eq!(
        ids_gpu, ids_ref,
        "Top-k ids mismatch for T={}, d_model={}, E={}, K={}, renorm={}",
        t, d_model, e, k, renorm
    );

    for i in 0..(t * k) {
        let diff = (probs_gpu[i] - probs_ref[i]).abs().to_f32().unwrap();

        // Tolerance accounts for numerical differences in:
        // 1. Router matmul: GPU uses vec4 FMA + SIMD reductions, CPU uses scalar loops
        //    With bf16 inputs and f32 accumulation, minor differences expected
        // 2. bf16 quantization on probs output buffer
        // 3. TopK selection (minor)
        // 4. Softmax computation when renorm=true (exp/div operations)
        let atol = if renorm {
            2e-2 // Normalized probabilities with bf16: softmax + bf16 quantization
        } else {
            5e-2 // Raw logits with bf16: vec4 accumulation + bf16 output quantization
        };
        assert!(
            diff <= atol,
            "Top-k prob mismatch at {}: gpu={} ref={} (diff={}, atol={}) with T={}, d_model={}, E={}, K={}, renorm={}",
            i,
            probs_gpu[i].to_f32().unwrap(),
            probs_ref[i].to_f32().unwrap(),
            diff,
            atol,
            t,
            d_model,
            e,
            k,
            renorm
        );
    }
}

#[uzu_test]
fn test_router_topk_fused_matches_reference() {
    let configs =
        [(1usize, 64usize, 32usize, 4usize), (2, 128, 64, 8), (4, 256, 128, 16), (8, 256, 256, 32), (1, 512, 512, 64)];
    let renorm_options = [false, true];

    for_each_non_cpu_backend!(|B| {
        for &(t, d_model, e, k) in &configs {
            for &renorm in &renorm_options {
                run_router_topk_once::<B, bf16>(t, d_model, e, k, renorm);
            }
        }
    });
}
