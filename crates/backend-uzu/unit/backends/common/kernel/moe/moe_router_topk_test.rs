use std::fmt::Debug;

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Backend, Encoder, Kernels, kernel::MoeRouterTopKKernel},
        cpu::Cpu,
    },
    common::helpers::create_context,
    data_type::DataType,
};

#[derive(Copy, Clone, Debug)]
enum RouterInput {
    Plain,
    Biased,
    Gemma4,
}

fn get_output<B: Backend, T: ArrayElement + Float>(
    input: &[T],
    weights: &[T],
    bias: Option<&[T]>,
    router_scale: Option<&[T]>,
    per_expert_scale: Option<&[T]>,
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
    renorm: bool,
    router_norm_epsilon: Option<f32>,
    router_input_scale: Option<f32>,
) -> (Vec<i32>, Vec<T>) {
    let ctx = create_context::<B>();

    let input_array = ctx.create_array_from(&[input.len()], input);
    let weights_array = ctx.create_array_from(&[weights.len()], weights);
    let bias_array = bias.map(|bias| ctx.create_array_from(&[bias.len()], bias));
    let router_scale_array =
        router_scale.map(|router_scale| ctx.create_array_from(&[router_scale.len()], router_scale));
    let per_expert_scale_array =
        per_expert_scale.map(|per_expert_scale| ctx.create_array_from(&[per_expert_scale.len()], per_expert_scale));
    let mut ids = ctx.create_array_uninitialized(&[t * k], DataType::I32).into_allocation();
    let mut probs = ctx.create_array_uninitialized(&[t * k], T::data_type()).into_allocation();

    let kernel = <<B as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(
        &ctx,
        T::data_type(),
        bias_array.is_some(),
        router_scale_array.is_some(),
        per_expert_scale_array.is_some(),
        router_input_scale.is_some(),
        router_norm_epsilon.is_some(),
    )
    .expect("kernel");
    let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_array.allocation(),
        weights_array.allocation(),
        bias_array.as_ref().map(|bias| bias.allocation()),
        router_scale_array.as_ref().map(|router_scale| router_scale.allocation()),
        per_expert_scale_array.as_ref().map(|per_expert_scale| per_expert_scale.allocation()),
        &mut ids,
        &mut probs,
        t as u32,
        d_model as u32,
        e as u32,
        k as u32,
        renorm,
        router_norm_epsilon,
        router_input_scale,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (crate::common::helpers::allocation_to_vec(&ids), crate::common::helpers::allocation_to_vec(&probs))
}

fn run_router_topk_once<B: Backend, T: ArrayElement + Debug + Float>(
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
    renorm: bool,
    router_input: RouterInput,
) {
    assert_eq!(d_model % 4, 0, "router_topk kernel processes d_model in 4-wide chunks");

    let mut rng = StdRng::seed_from_u64(1234);
    let gemma4 = matches!(router_input, RouterInput::Gemma4);
    let input: Vec<T> = (0..t * d_model).map(|_| T::from(rng.random_range(-1.0..1.0)).unwrap()).collect();
    let weight_range = if gemma4 {
        -0.4..0.4
    } else {
        -1.0..1.0
    };
    let weight: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(weight_range.clone())).unwrap()).collect();
    let bias: Option<Vec<T>> = matches!(router_input, RouterInput::Biased)
        .then(|| (0..e).map(|_| T::from(rng.random_range(-0.5..0.5)).unwrap()).collect());
    let router_scale: Option<Vec<T>> =
        gemma4.then(|| (0..d_model).map(|i| T::from(0.75 + (i % 7) as f32 * 0.05).unwrap()).collect());
    let per_expert_scale: Option<Vec<T>> =
        gemma4.then(|| (0..e).map(|i| T::from(0.5 + (i % 11) as f32 * 0.03).unwrap()).collect());
    let router_norm_epsilon = gemma4.then_some(1e-6);
    let router_input_scale = gemma4.then_some((d_model as f32).sqrt().recip());

    let (ids_ref, probs_ref) = get_output::<Cpu, T>(
        input.as_slice(),
        weight.as_slice(),
        bias.as_deref(),
        router_scale.as_deref(),
        per_expert_scale.as_deref(),
        t,
        d_model,
        e,
        k,
        renorm,
        router_norm_epsilon,
        router_input_scale,
    );
    let (ids_gpu, probs_gpu) = get_output::<B, T>(
        input.as_slice(),
        weight.as_slice(),
        bias.as_deref(),
        router_scale.as_deref(),
        per_expert_scale.as_deref(),
        t,
        d_model,
        e,
        k,
        renorm,
        router_norm_epsilon,
        router_input_scale,
    );
    assert_eq!(
        ids_gpu, ids_ref,
        "Top-k ids mismatch for T={}, d_model={}, E={}, K={}, renorm={}, router_input={:?}",
        t, d_model, e, k, renorm, router_input
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
            "Top-k prob mismatch at {}: gpu={} ref={} (diff={}, atol={}) with T={}, d_model={}, E={}, K={}, renorm={}, router_input={:?}",
            i,
            probs_gpu[i].to_f32().unwrap(),
            probs_ref[i].to_f32().unwrap(),
            diff,
            atol,
            t,
            d_model,
            e,
            k,
            renorm,
            router_input
        );
    }
}

#[uzu_test]
fn test_router_topk_fused_matches_reference() {
    let configs: &[(usize, usize, usize, usize, RouterInput, &[bool])] = &[
        (1, 64, 32, 4, RouterInput::Plain, &[false, true]),
        (4, 256, 128, 16, RouterInput::Plain, &[false, true]),
        (1, 64, 32, 4, RouterInput::Biased, &[false, true]),
        (2, 128, 64, 8, RouterInput::Biased, &[false, true]),
        (4, 256, 128, 16, RouterInput::Biased, &[false, true]),
        (8, 256, 256, 32, RouterInput::Biased, &[false, true]),
        (1, 512, 512, 64, RouterInput::Biased, &[false, true]),
        (3, 256, 128, 8, RouterInput::Gemma4, &[true]),
        (1, 2816, 128, 8, RouterInput::Gemma4, &[true]),
    ];

    for_each_non_cpu_backend!(|B| {
        for &(t, d_model, e, k, router_input, renorm_options) in configs {
            for &renorm in renorm_options {
                run_router_topk_once::<B, bf16>(t, d_model, e, k, renorm, router_input);
            }
        }
    });
}
