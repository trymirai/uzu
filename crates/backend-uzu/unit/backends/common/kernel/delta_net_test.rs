#![cfg(metal_backend)]

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{
                Conv1dPackKernel, DeltaNetChunkedBuildWUKernel, DeltaNetChunkedCumsumKernel,
                DeltaNetChunkedFusedApplyKernel, DeltaNetChunkedGramKernel, DeltaNetChunkedMegaApplyKernel,
                DeltaNetChunkedPrepKernel, DeltaNetChunkedSolveKernel, DeltaNetChunkedSolveTKernel,
                DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel, DeltaNetPrefillKernel,
                DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
            },
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
};

fn run_conv_update<B: Backend>(
    in_proj: &[f32],
    w: &[f32],
    b: &[f32],
    state: &[f32],
    kernel_size: u32,
    conv_dim: u32,
    state_stride: u32,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create context");

    let w_array = context.create_array_from(&[w.len()], w);
    let b_array = context.create_array_from(&[b.len()], b);
    let mut in_out = context.create_array_from(&[in_proj.len()], in_proj).into_allocation();
    let mut state_allocation = context.create_array_from(&[state.len()], state).into_allocation();

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(&context, DataType::F32, true)
        .expect("Failed to create kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        w_array.allocation(),
        Some(b_array.allocation()),
        &mut in_out,
        &mut state_allocation,
        kernel_size,
        conv_dim,
        state_stride,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = crate::tests::helpers::allocation_prefix_to_vec::<B, f32>(&in_out, conv_dim as usize);
    let new_state = crate::tests::helpers::allocation_to_vec::<B, f32>(&state_allocation);
    (out, new_state)
}

fn run_delta_net_update<B: Backend>(
    in_proj: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_weight: &[f32],
    state: &[f32],
    num_v_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create context");

    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias);
    let norm_weight_array = context.create_array_from(&[norm_weight.len()], norm_weight);
    let mut state_allocation = context.create_array_from(&[state.len()], state).into_allocation();
    let mut out = context.create_array_zeros(&[value_dim as usize], DataType::F32).into_allocation();

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetUpdateKernel::new(&context, DataType::F32, head_k_dim)
        .expect("Failed to create kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        in_proj_array.allocation(),
        a_log_array.allocation(),
        dt_bias_array.allocation(),
        norm_weight_array.allocation(),
        &mut state_allocation,
        &mut out,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        key_dim,
        value_dim,
        1e-6f32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = crate::tests::helpers::allocation_to_vec::<B, f32>(&out);
    let new_state = crate::tests::helpers::allocation_to_vec::<B, f32>(&state_allocation);
    (out, new_state)
}

fn assert_close(
    a: &[f32],
    b: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch: {} vs {}", a.len(), b.len());
    assert!(
        is_close::default().abs_tol(atol).rel_tol(rtol).all_close(a.iter().copied(), b.iter().copied()),
        "{label}: values not close enough (atol={atol:.6e}, rtol={rtol:.6e})"
    );
}

// DeltaNetConvUpdate

#[uzu_test]
fn test_delta_net_conv_update_small() {
    let conv_dim = 32;
    let kernel_size = 4;
    let tap_count = kernel_size - 1;

    let in_proj: Vec<f32> = (0..conv_dim).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = (0..conv_dim * kernel_size).map(|i| ((i % 11) as f32) * 0.05 - 0.2).collect();
    let b: Vec<f32> = (0..conv_dim).map(|i| ((i % 5) as f32) * 0.01).collect();
    let state: Vec<f32> = (0..conv_dim * tap_count).map(|i| ((i % 13) as f32) * 0.02 - 0.1).collect();

    let (cpu_out, cpu_state) =
        run_conv_update::<Cpu>(&in_proj, &w, &b, &state, kernel_size as u32, conv_dim as u32, tap_count as u32);
    let (gpu_out, gpu_state) =
        run_conv_update::<Metal>(&in_proj, &w, &b, &state, kernel_size as u32, conv_dim as u32, tap_count as u32);

    assert_close(&cpu_out, &gpu_out, 1e-4, 1e-3, "ConvUpdate output");
    assert_close(&cpu_state, &gpu_state, 1e-5, 1e-4, "ConvUpdate state");
}

// DeltaNetConvScan

#[uzu_test]
fn test_delta_net_conv_scan() {
    let conv_dim = 32;
    let kernel_size: usize = 4;
    let tap_count = kernel_size - 1;
    let suffix_len: usize = 16;
    let extra_dim = 10;
    let total_proj_dim = conv_dim + extra_dim;

    let w: Vec<f32> = (0..conv_dim * kernel_size).map(|i| ((i % 11) as f32) * 0.05 - 0.2).collect();
    let b: Vec<f32> = (0..conv_dim).map(|i| ((i % 5) as f32) * 0.01).collect();
    let init_state: Vec<f32> = (0..conv_dim * tap_count).map(|i| ((i % 13) as f32) * 0.02 - 0.1).collect();
    let in_proj: Vec<bf16> =
        (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 37) as f32) * 0.02 - 0.3)).collect();
    let in_proj_f32: Vec<f32> = in_proj.iter().copied().map(f32::from).collect();

    // Reference: decode conv token-by-token
    let mut ref_state = init_state.clone();
    let mut ref_outputs = vec![0.0f32; suffix_len * conv_dim];
    for t in 0..suffix_len {
        let token_in: Vec<f32> = in_proj_f32[t * total_proj_dim..t * total_proj_dim + conv_dim].to_vec();
        let (out, new_state) = run_conv_update::<Cpu>(
            &token_in,
            &w,
            &b,
            &ref_state,
            kernel_size as u32,
            conv_dim as u32,
            tap_count as u32,
        );
        ref_state = new_state;
        ref_outputs[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(&out);
    }
    let ref_outputs: Vec<f32> = ref_outputs.iter().copied().map(|value| f32::from(bf16::from_f32(value))).collect();

    // Test: Conv1dPack + DeltaNetConvScan on Metal
    let context = <Metal as Backend>::Context::new().expect("context");
    let state_array = context.create_array_from(&[init_state.len()], &init_state);
    let mut in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj).into_allocation();
    let w_array = context.create_array_from(&[w.len()], &w);
    let b_array = context.create_array_from(&[b.len()], &b);

    let padded_len = (tap_count + suffix_len) * total_proj_dim;
    let mut padded_array = context.create_array_zeros(&[padded_len], DataType::F32).into_allocation();
    let mut state_out_array = context.create_array_zeros(&[conv_dim * tap_count], DataType::F32).into_allocation();

    let pack_kernel =
        <<Metal as Backend>::Kernels as Kernels>::Conv1dPackKernel::new(&context, DataType::F32, DataType::BF16)
            .expect("pack");
    let scan_kernel =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetConvScanKernel::new(&context, DataType::BF16, true)
            .expect("scan");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    pack_kernel.encode(
        state_array.allocation(),
        &in_proj_array,
        &mut padded_array,
        tap_count as u32,
        total_proj_dim as u32,
        suffix_len as u32,
        conv_dim as u32,
        &mut encoder,
    );
    scan_kernel.encode(
        &padded_array,
        w_array.allocation(),
        Some(b_array.allocation()),
        &mut in_proj_array,
        &mut state_out_array,
        suffix_len as u32,
        kernel_size as u32,
        total_proj_dim as u32,
        tap_count as u32,
        conv_dim as u32,
        total_proj_dim as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let in_proj_result: Vec<bf16> = crate::tests::helpers::allocation_to_vec(&in_proj_array);
    let in_proj_result: Vec<f32> = in_proj_result.into_iter().map(f32::from).collect();
    let mut scan_outputs = vec![0.0f32; suffix_len * conv_dim];
    for t in 0..suffix_len {
        scan_outputs[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&in_proj_result[t * total_proj_dim..t * total_proj_dim + conv_dim]);
    }
    let scan_state: Vec<f32> = crate::tests::helpers::allocation_to_vec(&state_out_array);

    assert_close(&ref_outputs, &scan_outputs, 1e-4, 1e-3, "ConvScan output");
    assert_close(&ref_state, &scan_state, 1e-5, 1e-4, "ConvScan state");
}

// DeltaNetUpdate (decode)

fn test_delta_net_update_impl(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    label: &str,
) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_k_dim * head_v_dim;

    let in_proj: Vec<f32> = (0..total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.05).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.3 + (i as f32) * 0.02).collect();
    let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();
    let state: Vec<f32> = (0..state_size).map(|i| ((i % 29) as f32) * 0.005 - 0.05).collect();

    let (cpu_out, cpu_state) = run_delta_net_update::<Cpu>(
        &in_proj,
        &a_log,
        &dt_bias,
        &norm_weight,
        &state,
        num_v_heads as u32,
        num_k_heads as u32,
        head_k_dim as u32,
        head_v_dim as u32,
        key_dim as u32,
        value_dim as u32,
    );
    let (gpu_out, gpu_state) = run_delta_net_update::<Metal>(
        &in_proj,
        &a_log,
        &dt_bias,
        &norm_weight,
        &state,
        num_v_heads as u32,
        num_k_heads as u32,
        head_k_dim as u32,
        head_v_dim as u32,
        key_dim as u32,
        value_dim as u32,
    );

    assert_close(&cpu_out, &gpu_out, 1e-3, 1e-2, &format!("{label} output"));
    assert_close(&cpu_state, &gpu_state, 1e-4, 1e-3, &format!("{label} state"));
}

#[uzu_test]
fn test_delta_net_update_qwen35_shapes() {
    test_delta_net_update_impl(48, 16, 128, 128, "DeltaNetUpdate Qwen3.5");
}

// DeltaNetPrefill + NormGate

fn run_prefill_with_norm_gate_typed<T: ArrayElement>(
    in_proj: &[T],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_weight: &[f32],
    state: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    suffix_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let num_dv_groups = head_v_dim.div_ceil(16) as u32;

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias);
    let norm_weight_array = context.create_array_from(&[norm_weight.len()], norm_weight);
    let mut state_array = context.create_array_from(&[state.len()], state).into_allocation();
    let mut out_array = context.create_array_zeros(&[suffix_len * value_dim], T::data_type()).into_allocation();
    let mut q_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut k_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();

    let mut beta_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut decay_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        T::data_type(),
        head_k_dim as u32,
    )
    .unwrap();
    let prefill_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(
        &context,
        T::data_type(),
        head_k_dim as u32,
    )
    .unwrap();
    let norm_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, T::data_type()).unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    prep_k.encode(
        in_proj_array.allocation(),
        a_log_array.allocation(),
        dt_bias_array.allocation(),
        &mut q_norm_array,
        &mut k_norm_array,
        &mut beta_array,
        &mut decay_array,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    prefill_k.encode(
        &q_norm_array,
        &k_norm_array,
        &beta_array,
        &decay_array,
        in_proj_array.allocation(),
        &mut state_array,
        &mut out_array,
        num_v_heads as u32,
        num_k_heads as u32,
        head_v_dim as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        num_dv_groups,
        &mut encoder,
    );
    norm_k.encode(
        &mut out_array,
        in_proj_array.allocation(),
        norm_weight_array.allocation(),
        num_v_heads as u32,
        head_v_dim as u32,
        value_dim as u32,
        conv_dim as u32,
        total_proj_dim as u32,
        1e-6f32,
        suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out: Vec<T> = crate::tests::helpers::allocation_to_vec(&out_array);
    let out = out.into_iter().map(|value| value.to_f32().expect("output to f32")).collect();
    let state = crate::tests::helpers::allocation_to_vec(&state_array);
    (out, state)
}

fn test_prefill_norm_gate_impl<T: ArrayElement>(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    suffix_len: usize,
    output_atol: f32,
    output_rtol: f32,
    label: &str,
) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_k_dim * head_v_dim;

    let in_proj: Vec<f32> = (0..suffix_len * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();
    let in_proj_typed: Vec<T> = in_proj
        .iter()
        .copied()
        .map(|value| <T as num_traits::NumCast>::from(value).expect("input to activation dtype"))
        .collect();
    let reference_in_proj: Vec<f32> = in_proj_typed
        .iter()
        .copied()
        .map(|value| <f32 as num_traits::NumCast>::from(value).expect("activation dtype to f32"))
        .collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.05).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.3 + (i as f32) * 0.02).collect();
    let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();
    let state: Vec<f32> = (0..state_size).map(|i| ((i % 29) as f32) * 0.005 - 0.05).collect();

    // Reference: fused decode kernel token-by-token
    let mut ref_state = state.clone();
    let mut ref_outputs = vec![0.0f32; suffix_len * value_dim];
    for token_index in 0..suffix_len {
        let token_in: Vec<f32> =
            reference_in_proj[token_index * total_proj_dim..(token_index + 1) * total_proj_dim].to_vec();
        let (out, new_state) = run_delta_net_update::<Cpu>(
            &token_in,
            &a_log,
            &dt_bias,
            &norm_weight,
            &ref_state,
            num_v_heads as u32,
            num_k_heads as u32,
            head_k_dim as u32,
            head_v_dim as u32,
            key_dim as u32,
            value_dim as u32,
        );
        ref_state = new_state;
        ref_outputs[token_index * value_dim..(token_index + 1) * value_dim].copy_from_slice(&out);
    }

    let (gpu_out, gpu_state) = run_prefill_with_norm_gate_typed(
        &in_proj_typed,
        &a_log,
        &dt_bias,
        &norm_weight,
        &state,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        suffix_len,
    );

    assert_close(&ref_outputs, &gpu_out, output_atol, output_rtol, &format!("{label} output"));
    assert_close(&ref_state, &gpu_state, 1e-3, 1e-2, &format!("{label} state"));
}

#[uzu_test]
fn test_delta_net_prefill_qwen35_shapes() {
    test_prefill_norm_gate_impl::<f32>(48, 16, 128, 128, 32, 1e-3, 1e-2, "Prefill+NormGate Qwen3.5");
}

#[uzu_test]
fn test_delta_net_prefill_qwen35_shapes_bf16() {
    test_prefill_norm_gate_impl::<bf16>(48, 16, 128, 128, 32, 2e-2, 5e-2, "Prefill+NormGate Qwen3.5 BF16");
}

#[uzu_test]
fn test_delta_net_prefill_prep() {
    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let suffix_len = 8usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;

    let in_proj: Vec<f32> = (0..suffix_len * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.05).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.3 + (i as f32) * 0.02).collect();

    // CPU reference via Kernels trait
    let cpu_ctx = <Cpu as Backend>::Context::new().expect("cpu context");
    let cpu_in_proj = cpu_ctx.create_array_from(&[in_proj.len()], &in_proj);
    let cpu_a_log = cpu_ctx.create_array_from(&[a_log.len()], &a_log);
    let cpu_dt_bias = cpu_ctx.create_array_from(&[dt_bias.len()], &dt_bias);
    let mut cpu_q = cpu_ctx.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut cpu_k = cpu_ctx.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut cpu_beta = cpu_ctx.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut cpu_decay = cpu_ctx.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();

    let cpu_prep = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &cpu_ctx,
        DataType::F32,
        head_k_dim as u32,
    )
    .unwrap();
    let mut cpu_enc = Encoder::new(cpu_ctx.as_ref()).expect("encoder");
    cpu_prep.encode(
        cpu_in_proj.allocation(),
        cpu_a_log.allocation(),
        cpu_dt_bias.allocation(),
        &mut cpu_q,
        &mut cpu_k,
        &mut cpu_beta,
        &mut cpu_decay,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut cpu_enc,
    );
    cpu_enc.end_encoding().submit().wait_until_completed().unwrap();

    let ref_q: Vec<f32> = crate::tests::helpers::allocation_to_vec(&cpu_q);
    let ref_k: Vec<f32> = crate::tests::helpers::allocation_to_vec(&cpu_k);
    let ref_beta: Vec<f32> = crate::tests::helpers::allocation_to_vec(&cpu_beta);
    let ref_decay: Vec<f32> = crate::tests::helpers::allocation_to_vec(&cpu_decay);

    // Metal
    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
    let mut q_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut k_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();

    let mut beta_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut decay_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::F32,
        head_k_dim as u32,
    )
    .unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        in_proj_array.allocation(),
        a_log_array.allocation(),
        dt_bias_array.allocation(),
        &mut q_norm_array,
        &mut k_norm_array,
        &mut beta_array,
        &mut decay_array,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let gpu_q: Vec<f32> = crate::tests::helpers::allocation_to_vec(&q_norm_array);
    let gpu_k: Vec<f32> = crate::tests::helpers::allocation_to_vec(&k_norm_array);
    let gpu_beta: Vec<f32> = crate::tests::helpers::allocation_to_vec(&beta_array);
    let gpu_decay: Vec<f32> = crate::tests::helpers::allocation_to_vec(&decay_array);

    assert_close(&gpu_q, &ref_q, 1e-4, 1e-3, "prep q_norm");
    assert_close(&gpu_k, &ref_k, 1e-4, 1e-3, "prep k_norm");
    assert_close(&gpu_beta, &ref_beta, 1e-4, 1e-3, "prep beta");
    assert_close(&gpu_decay, &ref_decay, 1e-4, 1e-3, "prep decay");
}

#[uzu_test]
#[ignore]
fn bench_delta_net_prefill() {
    use test_runner::perf::run_perf_with_warmup;

    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let suffix_len = 32usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_k_dim * head_v_dim;

    let in_proj: Vec<f32> = (0..suffix_len * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.05).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.3 + (i as f32) * 0.02).collect();
    let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
    let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight);
    let mut out_array = context.create_array_zeros(&[suffix_len * value_dim], DataType::F32).into_allocation();
    let mut q_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut k_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();

    let mut beta_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut decay_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();

    let num_dv_groups = head_v_dim.div_ceil(16) as u32;

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::F32,
        head_k_dim as u32,
    )
    .unwrap();
    let prefill_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(
        &context,
        DataType::F32,
        head_k_dim as u32,
    )
    .unwrap();
    let norm_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::F32).unwrap();

    eprintln!("\n=== DeltaNet Prefill Benchmark (Qwen3.5 shapes) ===");
    eprintln!(
        "  heads={num_v_heads} k_heads={num_k_heads} head_k={head_k_dim} head_v={head_v_dim} tokens={suffix_len}"
    );
    eprintln!("  state_size={state_size} ({:.2} MB)", state_size as f64 * 4.0 / 1024.0 / 1024.0);

    let prep_result = run_perf_with_warmup("prep_only", 5, 50, || {
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        prep_k.encode(
            in_proj_array.allocation(),
            a_log_array.allocation(),
            dt_bias_array.allocation(),
            &mut q_norm_array,
            &mut k_norm_array,
            &mut beta_array,
            &mut decay_array,
            num_v_heads as u32,
            num_k_heads as u32,
            key_dim as u32,
            value_dim as u32,
            suffix_len as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });
    prep_result.print();

    // Benchmark prep + prefill + norm_gate (production path)
    let mut state_array = context.create_array_zeros(&[state_size], DataType::F32).into_allocation();

    let prefill_result = run_perf_with_warmup("prep+prefill+norm_gate", 5, 50, || {
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        encoder.encode_fill(&mut state_array, 0);
        prep_k.encode(
            in_proj_array.allocation(),
            a_log_array.allocation(),
            dt_bias_array.allocation(),
            &mut q_norm_array,
            &mut k_norm_array,
            &mut beta_array,
            &mut decay_array,
            num_v_heads as u32,
            num_k_heads as u32,
            key_dim as u32,
            value_dim as u32,
            suffix_len as u32,
            &mut encoder,
        );
        prefill_k.encode(
            &q_norm_array,
            &k_norm_array,
            &beta_array,
            &decay_array,
            in_proj_array.allocation(),
            &mut state_array,
            &mut out_array,
            num_v_heads as u32,
            num_k_heads as u32,
            head_v_dim as u32,
            key_dim as u32,
            value_dim as u32,
            suffix_len as u32,
            num_dv_groups,
            &mut encoder,
        );
        norm_k.encode(
            &mut out_array,
            in_proj_array.allocation(),
            norm_weight_array.allocation(),
            num_v_heads as u32,
            head_v_dim as u32,
            value_dim as u32,
            conv_dim as u32,
            total_proj_dim as u32,
            1e-6f32,
            suffix_len as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });
    prefill_result.print();
}

// Per-kernel wall-clock breakdown of the FUSED prefill pipeline at T=4096.
// Times each shared precompute kernel (Prep, Cumsum, Gram, Solve,
// BuildWU) individually and the single fused apply dispatch
// (`DeltaNetChunkedFusedApply`, VT=32), each isolated in its own encoder and
// warmed for >=500 ms of wall clock before timing. Reports median ms per kernel
// and its share of the summed pipeline total, so we can see whether the fused
// apply or the precompute chain dominates end-to-end time.
#[uzu_test]
#[ignore]
fn bench_delta_net_fused_kernel_breakdown() {
    use std::time::{Duration, Instant};

    use test_runner::perf::run_perf_with_warmup;

    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let bv = 32usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    eprintln!("\n=== DeltaNet FUSED pipeline per-kernel breakdown ===");
    eprintln!("  C={chunk_size} HV={num_v_heads} HK={num_k_heads} K={head_k_dim} V={head_v_dim}");
    eprintln!("  fused apply = DeltaNetChunkedFusedApply VT=32; >=500ms warmup per kernel");
    eprintln!("FUSEDBRK\tT\tkernel\tstage\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");

    for suffix_len in [64usize, 256, 4096] {
        let num_chunks = suffix_len.div_ceil(chunk_size);
        eprintln!("\n--- FUSED pipeline per-kernel breakdown T={suffix_len} num_chunks={num_chunks} ---");

        let in_proj: Vec<bf16> =
            (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
        let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
        let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

        let context = <Metal as Backend>::Context::new().expect("context");
        let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
        let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight);

        let alloc = |data_type, len| context.create_array_zeros(&[len], data_type).into_allocation();

        let mut q_norm = alloc(DataType::F32, suffix_len * key_dim);
        let mut k_norm = alloc(DataType::F32, suffix_len * key_dim);
        let mut beta = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut log_decay = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut g = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut kk = alloc(DataType::F32, num_chunks * num_k_heads * chunk_size * chunk_size);
        let mut qk_scaled = alloc(DataType::F32, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut a_packed =
            alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size);
        let mut a_inv = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * block_size * block_size);
        let mut w = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * head_k_dim);
        let mut u = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * head_v_dim);
        let mut fused_state = alloc(DataType::F32, state_size);
        let mut fused_out = alloc(DataType::BF16, suffix_len * value_dim);

        let prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
        let gram_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, false).unwrap();
        let solve_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
        let build_wu_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedBuildWUKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            128,
            64,
            bv as u32,
            false,
            false,
        )
        .unwrap();
        let norm_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::BF16).unwrap();
        let fused_vt32_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedFusedApplyKernel::new(
            &context,
            DataType::BF16,
            32,
        )
        .unwrap();

        // Prime every intermediate buffer once so each isolated kernel reads valid,
        // representative inputs during timing.
        {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut fused_state, 0);
            prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut q_norm,
                &mut k_norm,
                &mut beta,
                &mut log_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, &mut encoder);
            gram_k.encode(
                &q_norm,
                &k_norm,
                &g,
                &mut kk,
                &mut qk_scaled,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            solve_k.encode(
                &kk,
                &beta,
                &g,
                &mut a_packed,
                &mut a_inv,
                num_v_heads as u32,
                num_k_heads as u32,
                suffix_len as u32,
                &mut encoder,
            );
            build_wu_k.encode(
                &k_norm,
                in_proj_array.allocation(),
                &beta,
                &g,
                &a_packed,
                &a_inv,
                &mut w,
                &mut u,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        }

        let iterations = 50usize;

        // Warm a closure for >=500 ms wall clock before timing it.
        let warmup = |f: &mut dyn FnMut()| {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(500) {
                f();
            }
        };

        let mut results: Vec<(&str, test_runner::perf::PerfResult)> = Vec::new();

        macro_rules! time_kernel {
        ($name:expr, |$enc:ident| $body:block) => {{
            let mut run = || {
                let mut $enc = Encoder::new(context.as_ref()).expect("encoder");
                $body
                $enc.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            let result = run_perf_with_warmup($name, 3, iterations, &mut run);
            eprintln!(
                "FUSEDBRK\t{}\t{}\tprecompute\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                suffix_len, $name, result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms
            );
            results.push(($name, result));
        }};
    }

        time_kernel!("prep", |encoder| {
            prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut q_norm,
                &mut k_norm,
                &mut beta,
                &mut log_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        time_kernel!("cumsum", |encoder| {
            cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, &mut encoder);
        });
        time_kernel!("gram", |encoder| {
            gram_k.encode(
                &q_norm,
                &k_norm,
                &g,
                &mut kk,
                &mut qk_scaled,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        time_kernel!("solve", |encoder| {
            solve_k.encode(
                &kk,
                &beta,
                &g,
                &mut a_packed,
                &mut a_inv,
                num_v_heads as u32,
                num_k_heads as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        time_kernel!("build_wu", |encoder| {
            build_wu_k.encode(
                &k_norm,
                in_proj_array.allocation(),
                &beta,
                &g,
                &a_packed,
                &a_inv,
                &mut w,
                &mut u,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        // The fused apply dispatch (its own stage label).
        {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                fused_vt32_k.encode(
                    &w,
                    &u,
                    &q_norm,
                    &k_norm,
                    &qk_scaled,
                    &g,
                    &mut fused_state,
                    &mut fused_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            let result = run_perf_with_warmup("fused_apply_vt32", 3, iterations, &mut run);
            eprintln!(
                "FUSEDBRK\t{}\t{}\tapply\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                suffix_len, "fused_apply_vt32", result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms
            );
            results.push(("fused_apply_vt32", result));
        }

        // NormGate (final activation) timed for completeness; not counted in the
        // precompute-vs-apply split but reported so the full end-to-end is visible.
        let norm_result = {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                norm_k.encode(
                    &mut fused_out,
                    in_proj_array.allocation(),
                    norm_weight_array.allocation(),
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            let result = run_perf_with_warmup("norm_gate", 3, iterations, &mut run);
            eprintln!(
                "FUSEDBRK\t{}\t{}\tnorm_gate\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                suffix_len, "norm_gate", result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms
            );
            result
        };

        // Grouped single-encoder measurements. Isolated per-kernel timing overcounts
        // (each kernel pays its own submit/wait boundary with no cross-kernel
        // pipelining), so these grouped runs give the in-context, end-to-end-matching
        // attribution: the precompute chain in one encoder, and the whole pipeline
        // (precompute + apply + norm) in one encoder as the real end-to-end number.
        macro_rules! encode_precompute {
            ($encoder:expr) => {{
                prep_k.encode(
                    in_proj_array.allocation(),
                    a_log_array.allocation(),
                    dt_bias_array.allocation(),
                    &mut q_norm,
                    &mut k_norm,
                    &mut beta,
                    &mut log_decay,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    $encoder,
                );
                cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, $encoder);
                gram_k.encode(
                    &q_norm,
                    &k_norm,
                    &g,
                    &mut kk,
                    &mut qk_scaled,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    suffix_len as u32,
                    $encoder,
                );
                solve_k.encode(
                    &kk,
                    &beta,
                    &g,
                    &mut a_packed,
                    &mut a_inv,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    suffix_len as u32,
                    $encoder,
                );
                build_wu_k.encode(
                    &k_norm,
                    in_proj_array.allocation(),
                    &beta,
                    &g,
                    &a_packed,
                    &a_inv,
                    &mut w,
                    &mut u,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    $encoder,
                );
            }};
        }

        let precompute_group = {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encode_precompute!(&mut encoder);
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            run_perf_with_warmup("precompute_chain", 3, iterations, &mut run)
        };

        let pipeline_group = {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encoder.encode_fill(&mut fused_state, 0);
                encode_precompute!(&mut encoder);
                fused_vt32_k.encode(
                    &w,
                    &u,
                    &q_norm,
                    &k_norm,
                    &qk_scaled,
                    &g,
                    &mut fused_state,
                    &mut fused_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                norm_k.encode(
                    &mut fused_out,
                    in_proj_array.allocation(),
                    norm_weight_array.allocation(),
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            run_perf_with_warmup("full_pipeline", 3, iterations, &mut run)
        };

        eprintln!(
            "FUSEDBRK\t{}\tprecompute_chain\tgroup\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            suffix_len,
            precompute_group.median_ms,
            precompute_group.min_ms,
            precompute_group.max_ms,
            precompute_group.std_dev_ms
        );
        eprintln!(
            "FUSEDBRK\t{}\tfull_pipeline\tgroup\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            suffix_len,
            pipeline_group.median_ms,
            pipeline_group.min_ms,
            pipeline_group.max_ms,
            pipeline_group.std_dev_ms
        );

        let apply_ms = results.iter().find(|(n, _)| *n == "fused_apply_vt32").unwrap().1.median_ms;
        let precompute_ms: f64 =
            results.iter().filter(|(n, _)| *n != "fused_apply_vt32").map(|(_, r)| r.median_ms).sum();
        let pipeline_ms = precompute_ms + apply_ms;
        let total_with_norm = pipeline_ms + norm_result.median_ms;

        eprintln!("\n--- FUSED pipeline breakdown (T={suffix_len}) ---");
        eprintln!("kernel               median_ms   %of_pipeline");
        let mut sorted: Vec<&(&str, test_runner::perf::PerfResult)> = results.iter().collect();
        sorted.sort_by(|a, b| b.1.median_ms.partial_cmp(&a.1.median_ms).unwrap());
        for (name, r) in sorted {
            eprintln!("  {:<18} {:8.4}    {:6.2}%", name, r.median_ms, 100.0 * r.median_ms / pipeline_ms);
        }
        eprintln!("  {:<18} {:8.4}", "(norm_gate)", norm_result.median_ms);
        eprintln!(
            "\nSUMMARY(isolated)\tprecompute={:.4}ms ({:.1}%)\tapply={:.4}ms ({:.1}%)\tpipeline={:.4}ms\ttotal+norm={:.4}ms",
            precompute_ms,
            100.0 * precompute_ms / pipeline_ms,
            apply_ms,
            100.0 * apply_ms / pipeline_ms,
            pipeline_ms,
            total_with_norm
        );

        // In-context (grouped single-encoder) attribution — this matches the real
        // end-to-end number. apply_in_context = full_pipeline - precompute_chain - norm.
        let e2e = pipeline_group.median_ms;
        let precompute_ctx = precompute_group.median_ms;
        let apply_ctx = (e2e - precompute_ctx - norm_result.median_ms).max(0.0);
        eprintln!(
            "SUMMARY(grouped)\te2e(full_pipeline)={:.4}ms\tprecompute_chain={:.4}ms ({:.1}%)\tapply_in_context={:.4}ms ({:.1}%)\tnorm={:.4}ms ({:.1}%)",
            e2e,
            precompute_ctx,
            100.0 * precompute_ctx / e2e,
            apply_ctx,
            100.0 * apply_ctx / e2e,
            norm_result.median_ms,
            100.0 * norm_result.median_ms / e2e
        );
    }
}

// ===========================================================================
// DeltaNetChunkedFusedApply correctness tests
// ===========================================================================

// Largest absolute and (denominator-guarded) relative error between two slices.
fn max_abs_rel_err(
    a: &[f32],
    b: &[f32],
) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let abs = (x - y).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        let denom = x.abs().max(y.abs());
        if denom > 1e-4 {
            let rel = abs / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    (max_abs, max_rel)
}

// Run the fused persistent chunk-scan kernel on backend `B`, uploading all
// inputs and reading back the final `out` (f32) and mutated `state` (f32).
#[allow(clippy::too_many_arguments)]
fn run_fused_apply<B: Backend>(
    w: &[bf16],
    u: &[bf16],
    q_norm: &[f32],
    k_norm: &[f32],
    qk_scaled: &[f32],
    g: &[f32],
    state: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    suffix_len: usize,
    vt: u32,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("context");
    let w_a = context.create_array_from(&[w.len()], w);
    let u_a = context.create_array_from(&[u.len()], u);
    let q_a = context.create_array_from(&[q_norm.len()], q_norm);
    let k_a = context.create_array_from(&[k_norm.len()], k_norm);
    let qk_a = context.create_array_from(&[qk_scaled.len()], qk_scaled);
    let g_a = context.create_array_from(&[g.len()], g);
    let mut state_alloc = context.create_array_from(&[state.len()], state).into_allocation();
    let mut out_alloc = context.create_array_zeros(&[suffix_len * value_dim], DataType::F32).into_allocation();

    let kernel =
        <<B as Backend>::Kernels as Kernels>::DeltaNetChunkedFusedApplyKernel::new(&context, DataType::F32, vt)
            .expect("fused kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode(
        w_a.allocation(),
        u_a.allocation(),
        q_a.allocation(),
        k_a.allocation(),
        qk_a.allocation(),
        g_a.allocation(),
        &mut state_alloc,
        &mut out_alloc,
        num_v_heads as u32,
        num_k_heads as u32,
        head_v_dim as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = crate::tests::helpers::allocation_to_vec::<B, f32>(&out_alloc);
    let new_state = crate::tests::helpers::allocation_to_vec::<B, f32>(&state_alloc);
    (out, new_state)
}

// Deterministic pseudo-random inputs for the fused kernel at qwen3.5 shapes.
// Returns (w, u, q_norm, k_norm, qk_scaled, g, state).
#[allow(clippy::type_complexity)]
fn make_fused_inputs(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
    suffix_len: usize,
) -> (Vec<bf16>, Vec<bf16>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let state_size = num_v_heads * head_v_dim * head_k_dim;
    let num_chunks = suffix_len.div_ceil(chunk_size);

    let w: Vec<bf16> = (0..num_chunks * num_v_heads * chunk_size * head_k_dim)
        .map(|i| bf16::from_f32((((i * 7 + 3) % 31) as f32 - 15.0) * 0.01))
        .collect();
    let u: Vec<bf16> = (0..num_chunks * num_v_heads * chunk_size * head_v_dim)
        .map(|i| bf16::from_f32((((i * 5 + 1) % 29) as f32 - 14.0) * 0.02))
        .collect();
    let q_norm: Vec<f32> = (0..suffix_len * key_dim).map(|i| (((i * 3 + 2) % 41) as f32 - 20.0) * 0.01).collect();
    let k_norm: Vec<f32> = (0..suffix_len * key_dim).map(|i| (((i * 11 + 7) % 37) as f32 - 18.0) * 0.01).collect();
    // qk_scaled is causal-masked in the real pipeline; mask here too so the
    // inputs stay in the same regime (the fused and mirror math is identical
    // either way, but this keeps values realistic).
    let mut qk_scaled = vec![0.0f32; num_chunks * num_v_heads * chunk_size * chunk_size];
    for chunk in 0..num_chunks {
        for hv in 0..num_v_heads {
            for row in 0..chunk_size {
                for col in 0..chunk_size {
                    let idx = ((chunk * num_v_heads + hv) * chunk_size + row) * chunk_size + col;
                    qk_scaled[idx] = if col <= row {
                        (((idx * 13 + 5) % 23) as f32 - 11.0) * 0.01
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    // g is a per-token cumulative log-decay (negative, reset per chunk).
    let mut g = vec![0.0f32; suffix_len * num_v_heads];
    for token in 0..suffix_len {
        let local = token % chunk_size;
        for hv in 0..num_v_heads {
            let step = 0.01 + ((hv % 7) as f32) * 0.002;
            g[token * num_v_heads + hv] = -(local as f32 + 1.0) * step;
        }
    }
    let state: Vec<f32> = (0..state_size).map(|i| (((i * 19 + 4) % 29) as f32) * 0.004 - 0.05).collect();

    (w, u, q_norm, k_norm, qk_scaled, g, state)
}

// Test 1: fused Metal kernel vs its CPU mirror (the kernel's own ground truth),
// driven by identical random inputs. Covers VT = 16 and VT = 32.
fn fused_vs_cpu_mirror_impl(
    suffix_len: usize,
    vt: u32,
) {
    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;

    let (w, u, q_norm, k_norm, qk_scaled, g, state) =
        make_fused_inputs(num_v_heads, num_k_heads, head_k_dim, head_v_dim, chunk_size, suffix_len);

    let (metal_out, metal_state) = run_fused_apply::<Metal>(
        &w,
        &u,
        &q_norm,
        &k_norm,
        &qk_scaled,
        &g,
        &state,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        key_dim,
        value_dim,
        suffix_len,
        vt,
    );
    let (cpu_out, cpu_state) = run_fused_apply::<Cpu>(
        &w,
        &u,
        &q_norm,
        &k_norm,
        &qk_scaled,
        &g,
        &state,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        key_dim,
        value_dim,
        suffix_len,
        vt,
    );

    let (out_abs, out_rel) = max_abs_rel_err(&metal_out, &cpu_out);
    let (state_abs, state_rel) = max_abs_rel_err(&metal_state, &cpu_state);
    eprintln!(
        "FUSED_VS_MIRROR T={suffix_len} VT={vt}: out max_abs={out_abs:.3e} max_rel={out_rel:.3e} \
         state max_abs={state_abs:.3e} max_rel={state_rel:.3e}"
    );

    // Same inputs, same math; only fragment-tiled f32 accumulation order and
    // fast::exp vs std exp differ. This is far tighter than the pipeline test.
    assert_close(&cpu_out, &metal_out, 5e-3, 5e-3, &format!("fused vs mirror out (T={suffix_len} VT={vt})"));
    assert_close(&cpu_state, &metal_state, 5e-3, 5e-3, &format!("fused vs mirror state (T={suffix_len} VT={vt})"));
}

#[uzu_test]
fn test_delta_net_chunked_fused_vs_cpu_mirror_vt16() {
    // Multi-chunk with a partial last chunk exercises the bounded edge paths.
    fused_vs_cpu_mirror_impl(130, 16);
}

#[uzu_test]
fn test_delta_net_chunked_fused_vs_cpu_mirror_vt32() {
    fused_vs_cpu_mirror_impl(130, 32);
}

// Runs the chunked precompute chain (Prep, Cumsum, Gram, Solve, BuildWU) on
// Metal and reads back the fused kernel inputs, including the precomputed
// chunk-local prefix g. The fused kernel folds the decay scale exp(g_last - g_t)
// on the fly, so no decay_scale buffer is produced. All independent of the
// recurrent state.
#[allow(clippy::type_complexity)]
fn run_chunked_precompute_metal(
    in_proj: &[bf16],
    a_log: &[f32],
    dt_bias: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
    suffix_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<bf16>, Vec<bf16>, Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);
    let bv = 32usize;

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias);

    let mut q_norm = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut k_norm = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut beta = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut log_decay = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut g = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut kk = context
        .create_array_zeros(&[num_chunks * num_k_heads * chunk_size * chunk_size], DataType::F32)
        .into_allocation();
    let mut qk_scaled = context
        .create_array_zeros(&[num_chunks * num_v_heads * chunk_size * chunk_size], DataType::F32)
        .into_allocation();
    let mut a_packed = context
        .create_array_zeros(
            &[num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size],
            DataType::F32,
        )
        .into_allocation();
    let mut a_inv = context
        .create_array_zeros(&[num_chunks * num_v_heads * num_blocks * block_size * block_size], DataType::F32)
        .into_allocation();
    let mut w = context
        .create_array_zeros(&[num_chunks * num_v_heads * chunk_size * head_k_dim], DataType::BF16)
        .into_allocation();
    let mut u = context
        .create_array_zeros(&[num_chunks * num_v_heads * chunk_size * head_v_dim], DataType::BF16)
        .into_allocation();

    let prep_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
            .unwrap();
    let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
    let gram_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, false).unwrap();
    let solve_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
    let build_wu_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedBuildWUKernel::new(
        &context,
        DataType::BF16,
        DataType::BF16,
        128,
        64,
        bv as u32,
        false,
        false,
    )
    .unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        in_proj_array.allocation(),
        a_log_array.allocation(),
        dt_bias_array.allocation(),
        &mut q_norm,
        &mut k_norm,
        &mut beta,
        &mut log_decay,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, &mut encoder);
    gram_k.encode(
        &q_norm,
        &k_norm,
        &g,
        &mut kk,
        &mut qk_scaled,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    solve_k.encode(
        &kk,
        &beta,
        &g,
        &mut a_packed,
        &mut a_inv,
        num_v_heads as u32,
        num_k_heads as u32,
        suffix_len as u32,
        &mut encoder,
    );
    build_wu_k.encode(
        &k_norm,
        in_proj_array.allocation(),
        &beta,
        &g,
        &a_packed,
        &a_inv,
        &mut w,
        &mut u,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (
        crate::tests::helpers::allocation_to_vec(&q_norm),
        crate::tests::helpers::allocation_to_vec(&k_norm),
        crate::tests::helpers::allocation_to_vec(&w),
        crate::tests::helpers::allocation_to_vec(&u),
        crate::tests::helpers::allocation_to_vec(&qk_scaled),
        crate::tests::helpers::allocation_to_vec(&g),
    )
}

// Recurrent reference (DeltaNetPrefillPrep + DeltaNetPrefill, no norm gate),
// starting from a given initial state. Produces raw f32 output and final state.
fn run_recurrent_raw(
    in_proj: &[bf16],
    a_log: &[f32],
    dt_bias: &[f32],
    init_state: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    suffix_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let num_dv_groups = head_v_dim.div_ceil(16) as u32;

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias);
    let mut state_array = context.create_array_from(&[init_state.len()], init_state).into_allocation();
    // The prefill/prep dtype must match the (bf16) in_proj activation dtype;
    // out is therefore bf16 (same as the existing chunked-vs-recurrent bench).
    let mut out_array = context.create_array_zeros(&[suffix_len * value_dim], DataType::BF16).into_allocation();
    let mut q_norm = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut k_norm = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut beta = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut decay = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::BF16,
        head_k_dim as u32,
    )
    .unwrap();
    let prefill_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(
        &context,
        DataType::BF16,
        head_k_dim as u32,
    )
    .unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        in_proj_array.allocation(),
        a_log_array.allocation(),
        dt_bias_array.allocation(),
        &mut q_norm,
        &mut k_norm,
        &mut beta,
        &mut decay,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    prefill_k.encode(
        &q_norm,
        &k_norm,
        &beta,
        &decay,
        in_proj_array.allocation(),
        &mut state_array,
        &mut out_array,
        num_v_heads as u32,
        num_k_heads as u32,
        head_v_dim as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        num_dv_groups,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out_bf16: Vec<bf16> = crate::tests::helpers::allocation_to_vec(&out_array);
    let out: Vec<f32> = out_bf16.iter().map(|value| value.to_f32()).collect();
    (out, crate::tests::helpers::allocation_to_vec(&state_array))
}

// Test 2: full fused pipeline (chunked precompute + DeltaNetChunkedFusedApply)
// vs the recurrent reference DeltaNetPrefill at qwen3.5 shapes. Compares the
// raw output (pre norm-gate) and the final state, for both VT variants and a
// zero and nonzero initial state.
fn fused_pipeline_vs_recurrent_impl(suffix_len: usize) {
    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;

    let in_proj: Vec<bf16> =
        (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();

    // Precompute (state-independent) once.
    let (q_norm, k_norm, w, u, qk_scaled, g) = run_chunked_precompute_metal(
        &in_proj,
        &a_log,
        &dt_bias,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
        suffix_len,
    );

    for nonzero_init in [false, true] {
        let init_state: Vec<f32> = if nonzero_init {
            (0..state_size).map(|i| (((i * 13 + 5) % 31) as f32) * 0.003 - 0.045).collect()
        } else {
            vec![0.0f32; state_size]
        };

        let (ref_out, ref_state) = run_recurrent_raw(
            &in_proj,
            &a_log,
            &dt_bias,
            &init_state,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            suffix_len,
        );

        for vt in [16u32, 32u32] {
            let (fused_out, fused_state) = run_fused_apply::<Metal>(
                &w,
                &u,
                &q_norm,
                &k_norm,
                &qk_scaled,
                &g,
                &init_state,
                num_v_heads,
                num_k_heads,
                head_v_dim,
                key_dim,
                value_dim,
                suffix_len,
                vt,
            );

            let (out_abs, out_rel) = max_abs_rel_err(&ref_out, &fused_out);
            let (state_abs, state_rel) = max_abs_rel_err(&ref_state, &fused_state);
            eprintln!(
                "FUSED_PIPELINE T={suffix_len} VT={vt} nonzero_init={nonzero_init}: \
                 out max_abs={out_abs:.3e} max_rel={out_rel:.3e} \
                 state max_abs={state_abs:.3e} max_rel={state_rel:.3e}"
            );

            let label = format!("fused pipeline T={suffix_len} VT={vt} nonzero_init={nonzero_init}");
            // Observed max errors across all T (incl. 4096, partial chunks and
            // nonzero init) are out abs <= 1e-5, state abs <= 1.3e-5; a real bug
            // (wrong indexing, partial-chunk mishandling, the old VT=16 MXU
            // pairing bug) produced errors >= 5e-2. These bounds sit in that gap:
            // ~100x tighter than the existing chunked-vs-recurrent test (5e-2),
            // yet ~80x above the noise floor so they are not flaky. The f32 state
            // path is held tighter than the bf16-limited output.
            assert_close(&ref_out, &fused_out, 2e-3, 5e-2, &format!("{label} out"));
            assert_close(&ref_state, &fused_state, 1e-3, 5e-2, &format!("{label} state"));
        }
    }
}

#[uzu_test]
fn test_delta_net_chunked_fused_pipeline_vs_recurrent() {
    for suffix_len in [1usize, 63, 64, 65, 200, 256, 4096] {
        fused_pipeline_vs_recurrent_impl(suffix_len);
    }
}

// ===========================================================================
// DeltaNetChunkedMegaApply (Mode L) correctness tests
// ===========================================================================

// Run the Mode L mega kernel on backend `B`, uploading all inputs and reading
// back the final `out` (f32) and mutated `state` (f32). in_proj is bf16 (the V
// slice source), T and A are bf16 device operands; state is f32.
#[allow(clippy::too_many_arguments)]
fn run_mega_apply<B: Backend>(
    q_norm: &[f32],
    k_norm: &[f32],
    in_proj: &[bf16],
    qk_scaled: &[f32],
    t_mat: &[bf16],
    g: &[f32],
    beta: &[f32],
    state: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    suffix_len: usize,
    vt: u32,
    use_mxu: bool,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("context");
    let q_a = context.create_array_from(&[q_norm.len()], q_norm);
    let k_a = context.create_array_from(&[k_norm.len()], k_norm);
    let in_a = context.create_array_from(&[in_proj.len()], in_proj);
    let qk_a = context.create_array_from(&[qk_scaled.len()], qk_scaled);
    let t_a = context.create_array_from(&[t_mat.len()], t_mat);
    let g_a = context.create_array_from(&[g.len()], g);
    let beta_a = context.create_array_from(&[beta.len()], beta);
    let mut state_alloc = context.create_array_from(&[state.len()], state).into_allocation();
    let mut out_alloc = context.create_array_zeros(&[suffix_len * value_dim], DataType::F32).into_allocation();

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        vt,
        use_mxu,
    )
    .expect("mega kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode(
        q_a.allocation(),
        k_a.allocation(),
        in_a.allocation(),
        qk_a.allocation(),
        t_a.allocation(),
        g_a.allocation(),
        beta_a.allocation(),
        &mut state_alloc,
        &mut out_alloc,
        num_v_heads as u32,
        num_k_heads as u32,
        head_v_dim as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = crate::tests::helpers::allocation_to_vec::<B, f32>(&out_alloc);
    let new_state = crate::tests::helpers::allocation_to_vec::<B, f32>(&state_alloc);
    (out, new_state)
}

// Deterministic pseudo-random inputs for the mega kernel at qwen3.5 shapes.
// Returns (q_norm, k_norm, in_proj, qk_scaled, t_mat, g, beta, state). t_mat is
// unit lower triangular (diagonal 1, strictly-lower random, upper 0) bf16.
#[allow(clippy::type_complexity)]
fn make_mega_inputs(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
    suffix_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<bf16>, Vec<f32>, Vec<bf16>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;
    let num_chunks = suffix_len.div_ceil(chunk_size);

    let q_norm: Vec<f32> = (0..suffix_len * key_dim).map(|i| (((i * 3 + 2) % 41) as f32 - 20.0) * 0.01).collect();
    let k_norm: Vec<f32> = (0..suffix_len * key_dim).map(|i| (((i * 11 + 7) % 37) as f32 - 18.0) * 0.01).collect();
    let in_proj: Vec<bf16> =
        (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32((((i * 5 + 1) % 29) as f32 - 14.0) * 0.01)).collect();

    // qk_scaled is causal-masked in the real pipeline; mask here too.
    let mut qk_scaled = vec![0.0f32; num_chunks * num_v_heads * chunk_size * chunk_size];
    for chunk in 0..num_chunks {
        for hv in 0..num_v_heads {
            for row in 0..chunk_size {
                for col in 0..chunk_size {
                    let idx = ((chunk * num_v_heads + hv) * chunk_size + row) * chunk_size + col;
                    qk_scaled[idx] = if col <= row {
                        (((idx * 13 + 5) % 23) as f32 - 11.0) * 0.01
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    // t_mat unit lower triangular (diagonal 1, strictly-lower small, upper 0).
    let mut t_mat = vec![bf16::from_f32(0.0); num_chunks * num_v_heads * chunk_size * chunk_size];
    for chunk in 0..num_chunks {
        for hv in 0..num_v_heads {
            for row in 0..chunk_size {
                for col in 0..chunk_size {
                    let idx = ((chunk * num_v_heads + hv) * chunk_size + row) * chunk_size + col;
                    let value = if col == row {
                        1.0f32
                    } else if col < row {
                        (((idx * 17 + 9) % 19) as f32 - 9.0) * 0.01
                    } else {
                        0.0
                    };
                    t_mat[idx] = bf16::from_f32(value);
                }
            }
        }
    }

    // g is a per-token cumulative log-decay (negative, reset per chunk).
    let mut g = vec![0.0f32; suffix_len * num_v_heads];
    for token in 0..suffix_len {
        let local = token % chunk_size;
        for hv in 0..num_v_heads {
            let step = 0.01 + ((hv % 7) as f32) * 0.002;
            g[token * num_v_heads + hv] = -(local as f32 + 1.0) * step;
        }
    }
    let beta: Vec<f32> = (0..suffix_len * num_v_heads).map(|i| 0.3 + (((i * 7 + 3) % 11) as f32) * 0.05).collect();
    let state: Vec<f32> = (0..state_size).map(|i| (((i * 19 + 4) % 29) as f32) * 0.004 - 0.05).collect();

    (q_norm, k_norm, in_proj, qk_scaled, t_mat, g, beta, state)
}

// Test 1: mega Metal kernel vs its CPU mirror, driven by identical random
// inputs. Covers VT = 16 and VT = 32.
fn mega_vs_cpu_mirror_impl(
    suffix_len: usize,
    vt: u32,
    use_mxu: bool,
) {
    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;

    let (q_norm, k_norm, in_proj, qk_scaled, t_mat, g, beta, state) =
        make_mega_inputs(num_v_heads, num_k_heads, head_k_dim, head_v_dim, chunk_size, suffix_len);

    let (metal_out, metal_state) = run_mega_apply::<Metal>(
        &q_norm,
        &k_norm,
        &in_proj,
        &qk_scaled,
        &t_mat,
        &g,
        &beta,
        &state,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        key_dim,
        value_dim,
        suffix_len,
        vt,
        use_mxu,
    );
    let (cpu_out, cpu_state) = run_mega_apply::<Cpu>(
        &q_norm,
        &k_norm,
        &in_proj,
        &qk_scaled,
        &t_mat,
        &g,
        &beta,
        &state,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        key_dim,
        value_dim,
        suffix_len,
        vt,
        use_mxu,
    );

    let (out_abs, out_rel) = max_abs_rel_err(&metal_out, &cpu_out);
    let (state_abs, state_rel) = max_abs_rel_err(&metal_state, &cpu_state);
    eprintln!(
        "MEGA_VS_MIRROR T={suffix_len} VT={vt}: out max_abs={out_abs:.3e} max_rel={out_rel:.3e} \
         state max_abs={state_abs:.3e} max_rel={state_rel:.3e}"
    );

    assert_close(&cpu_out, &metal_out, 5e-3, 5e-3, &format!("mega vs mirror out (T={suffix_len} VT={vt})"));
    assert_close(&cpu_state, &metal_state, 5e-3, 5e-3, &format!("mega vs mirror state (T={suffix_len} VT={vt})"));
}

#[uzu_test]
fn test_delta_net_chunked_mega_vs_cpu_mirror_vt16() {
    // VT=16 ships on the simdgroup backend.
    mega_vs_cpu_mirror_impl(130, 16, false);
}

#[uzu_test]
fn test_delta_net_chunked_mega_vs_cpu_mirror_vt32() {
    // VT=32 shipping config: MXU backend (unchanged default).
    mega_vs_cpu_mirror_impl(130, 32, true);
}

#[uzu_test]
fn test_delta_net_chunked_mega_vs_cpu_mirror_vt32_simd() {
    // NEW: VT=32 on the simdgroup backend (M1-M4 candidate, no MXU).
    mega_vs_cpu_mirror_impl(130, 32, false);
}

// Runs the Mode L precompute chain (Prep, Cumsum, Gram, Solve, SolveT) on Metal
// and reads back the mega kernel inputs, including the precomputed chunk-local
// prefix g, the dense bf16 inverse T, and beta. SolveT emits the dense T via a
// block forward substitution (identity RHS) over Solve's a_packed/a_inv block
// inverses. State-independent.
#[allow(clippy::type_complexity)]
fn run_mode_l_precompute_metal(
    in_proj: &[bf16],
    a_log: &[f32],
    dt_bias: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
    suffix_len: usize,
    use_mxu: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bf16>, Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj);
    let a_log_array = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias);

    let mut q_norm = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut k_norm = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32).into_allocation();
    let mut beta = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut log_decay = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut g = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32).into_allocation();
    let mut kk = context
        .create_array_zeros(&[num_chunks * num_k_heads * chunk_size * chunk_size], DataType::F32)
        .into_allocation();
    let mut qk_scaled = context
        .create_array_zeros(&[num_chunks * num_v_heads * chunk_size * chunk_size], DataType::F32)
        .into_allocation();
    let mut a_packed = context
        .create_array_zeros(
            &[num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size],
            DataType::F32,
        )
        .into_allocation();
    let mut a_inv = context
        .create_array_zeros(&[num_chunks * num_v_heads * num_blocks * block_size * block_size], DataType::F32)
        .into_allocation();
    let mut t_mat = context
        .create_array_zeros(&[num_chunks * num_v_heads * chunk_size * chunk_size], DataType::BF16)
        .into_allocation();

    let prep_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
            .unwrap();
    let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
    let gram_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, use_mxu).unwrap();
    let solve_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
    let solve_t_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32, use_mxu).unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        in_proj_array.allocation(),
        a_log_array.allocation(),
        dt_bias_array.allocation(),
        &mut q_norm,
        &mut k_norm,
        &mut beta,
        &mut log_decay,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, &mut encoder);
    gram_k.encode(
        &q_norm,
        &k_norm,
        &g,
        &mut kk,
        &mut qk_scaled,
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    solve_k.encode(
        &kk,
        &beta,
        &g,
        &mut a_packed,
        &mut a_inv,
        num_v_heads as u32,
        num_k_heads as u32,
        suffix_len as u32,
        &mut encoder,
    );
    solve_t_k.encode(&a_packed, &a_inv, &mut t_mat, num_v_heads as u32, suffix_len as u32, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (
        crate::tests::helpers::allocation_to_vec(&q_norm),
        crate::tests::helpers::allocation_to_vec(&k_norm),
        crate::tests::helpers::allocation_to_vec(&qk_scaled),
        crate::tests::helpers::allocation_to_vec(&t_mat),
        crate::tests::helpers::allocation_to_vec(&g),
        crate::tests::helpers::allocation_to_vec(&beta),
    )
}

// Test 2: full Mode L pipeline (prep, cumsum, gram, solveT + mega apply) vs the
// recurrent reference DeltaNetPrefill at qwen3.5 shapes. Compares raw output
// (pre norm-gate) and final state, for both VT variants and zero + nonzero
// initial state. SAME tolerances as the fused pipeline test.
fn mode_l_pipeline_vs_recurrent_impl(suffix_len: usize) {
    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;

    let in_proj: Vec<bf16> =
        (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();

    // Two precompute configs: simdgroup gram/solveT (config B, USE_MXU=false) and
    // bf16-operand MXU gram/solveT (config A, USE_MXU=true). Precompute is
    // VT-independent; megaApply VT=16 is simdgroup, VT=32 is MXU.
    let precompute = |use_mxu: bool| {
        run_mode_l_precompute_metal(
            &in_proj,
            &a_log,
            &dt_bias,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            suffix_len,
            use_mxu,
        )
    };
    let precompute_simd = precompute(false);
    let precompute_mxu = precompute(true);

    for nonzero_init in [false, true] {
        let init_state: Vec<f32> = if nonzero_init {
            (0..state_size).map(|i| (((i * 13 + 5) % 31) as f32) * 0.003 - 0.045).collect()
        } else {
            vec![0.0f32; state_size]
        };

        let (ref_out, ref_state) = run_recurrent_raw(
            &in_proj,
            &a_log,
            &dt_bias,
            &init_state,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            suffix_len,
        );

        // (config label, precompute tuple, mega VT, mega USE_MXU). Config B =
        // simd precompute + megaApply VT16 simd; Config A = mxu precompute +
        // megaApply VT32 MXU (shipping M5 default). "simd_vt32" is the NEW
        // decoupled variant: VT32 on the simdgroup backend (the M1-M4 candidate).
        // Crosses included for coverage. Measured: rounding gram/solveT operands
        // to bf16 (config A) leaves out/state abs error at ~1e-5 -- far under the
        // shipped tolerance -- so ALL configs assert at the same shipped tolerance
        // (out atol 2e-3, state atol 1e-3, rtol 5e-2).
        let configs: [(&str, &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bf16>, Vec<f32>, Vec<f32>), u32, bool); 4] = [
            ("B_simd", &precompute_simd, 16, false),
            ("simd_vt32", &precompute_simd, 32, false),
            ("mxu_vt16", &precompute_mxu, 16, false),
            ("A_mxu", &precompute_mxu, 32, true),
        ];

        for (cfg, pre, vt, mega_use_mxu) in configs {
            let (q_norm, k_norm, qk_scaled, t_mat, g, beta) = pre;
            let (mega_out, mega_state) = run_mega_apply::<Metal>(
                q_norm,
                k_norm,
                &in_proj,
                qk_scaled,
                t_mat,
                g,
                beta,
                &init_state,
                num_v_heads,
                num_k_heads,
                head_v_dim,
                key_dim,
                value_dim,
                suffix_len,
                vt,
                mega_use_mxu,
            );

            let (out_abs, out_rel) = max_abs_rel_err(&ref_out, &mega_out);
            let (state_abs, state_rel) = max_abs_rel_err(&ref_state, &mega_state);
            eprintln!(
                "MODE_L_PIPELINE cfg={cfg} T={suffix_len} VT={vt} nonzero_init={nonzero_init}: \
                 out max_abs={out_abs:.3e} max_rel={out_rel:.3e} \
                 state max_abs={state_abs:.3e} max_rel={state_rel:.3e}"
            );

            let label = format!("mode L pipeline cfg={cfg} T={suffix_len} VT={vt} nonzero_init={nonzero_init}");
            assert_close(&ref_out, &mega_out, 2e-3, 5e-2, &format!("{label} out"));
            assert_close(&ref_state, &mega_state, 1e-3, 5e-2, &format!("{label} state"));
        }
    }
}

#[uzu_test]
fn test_delta_net_chunked_mode_l_pipeline_vs_recurrent() {
    for suffix_len in [1usize, 63, 64, 65, 200, 256, 4096] {
        mode_l_pipeline_vs_recurrent_impl(suffix_len);
    }
}

// End-to-end prefill sweep comparing three paths at the SAME scope (precompute +
// apply + NormGate): the recurrent baseline, the Phase-A fused-7 pipeline
// (prep, cumsum, gram, solve, buildWU, fusedApply, normGate), and the Mode L
// mega pipeline (prep, cumsum, gram, solveT, megaApply, normGate; 6 dispatches,
// VT=32) — the headline chunked path. Each path warmed >=500 ms before timing.
// This is the canonical end-to-end sweep: recurrent vs Mode-L(VT32) is the
// primary comparison, with fused-7 retained as an optional column.
#[uzu_test]
#[ignore]
fn bench_delta_net_fused_vs_recurrent_prefill() {
    use std::time::{Duration, Instant};

    use test_runner::perf::run_perf_with_warmup;

    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let bv = 32usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;
    let num_dv_groups = head_v_dim.div_ceil(16);
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    eprintln!("\n=== DeltaNet Mode L vs Fused-7 vs Recurrent (end-to-end prefill) ===");
    eprintln!("  C={chunk_size} HV={num_v_heads} HK={num_k_heads} K={head_k_dim} V={head_v_dim} dtype=bf16");
    eprintln!("  scope: precompute + apply + NormGate for every path");
    eprintln!("MODELBENCH\tT\tpath\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");

    for suffix_len in [64usize, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768] {
        let num_chunks = suffix_len.div_ceil(chunk_size);

        let in_proj: Vec<bf16> =
            (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
        let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
        let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

        let context = <Metal as Backend>::Context::new().expect("context");
        let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
        let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight);

        let alloc = |data_type, len| context.create_array_zeros(&[len], data_type).into_allocation();

        // Recurrent buffers.
        let mut recurrent_out = alloc(DataType::BF16, suffix_len * value_dim);
        let mut recurrent_q_norm = alloc(DataType::F32, suffix_len * key_dim);
        let mut recurrent_k_norm = alloc(DataType::F32, suffix_len * key_dim);
        let mut recurrent_beta = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut recurrent_decay = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut recurrent_state = alloc(DataType::F32, state_size);

        // Shared precompute buffers.
        let mut q_norm_f = alloc(DataType::F32, suffix_len * key_dim);
        let mut k_norm_f = alloc(DataType::F32, suffix_len * key_dim);
        let mut beta_f = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut log_decay_f = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut g_f = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut kk_f = alloc(DataType::F32, num_chunks * num_k_heads * chunk_size * chunk_size);
        let mut qk_scaled_f = alloc(DataType::F32, num_chunks * num_v_heads * chunk_size * chunk_size);

        // Fused-7 only buffers.
        let mut a_packed_f =
            alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size);
        let mut a_inv_f = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * block_size * block_size);
        let mut w_f = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * head_k_dim);
        let mut u_f = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * head_v_dim);
        let mut fused_state = alloc(DataType::F32, state_size);
        let mut fused_out = alloc(DataType::BF16, suffix_len * value_dim);

        // Mode L only buffers (separate precompute set so its closure does not
        // share &mut with the fused closure).
        let mut q_norm_m = alloc(DataType::F32, suffix_len * key_dim);
        let mut k_norm_m = alloc(DataType::F32, suffix_len * key_dim);
        let mut beta_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut log_decay_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut g_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut kk_m = alloc(DataType::F32, num_chunks * num_k_heads * chunk_size * chunk_size);
        let mut qk_scaled_m = alloc(DataType::F32, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut a_packed_m =
            alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size);
        let mut a_inv_m = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * block_size * block_size);
        let mut t_mat_f = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut mode_l_state = alloc(DataType::F32, state_size);
        let mut mode_l_out = alloc(DataType::BF16, suffix_len * value_dim);

        // Kernels.
        let recurrent_prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let recurrent_prefill_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let norm_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::BF16).unwrap();
        let prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
        let gram_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, false).unwrap();
        let solve_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
        let build_wu_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedBuildWUKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            128,
            64,
            bv as u32,
            false,
            false,
        )
        .unwrap();
        let fused_vt32_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedFusedApplyKernel::new(
            &context,
            DataType::BF16,
            32,
        )
        .unwrap();
        let solve_t_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32, false)
                .unwrap();
        let mega_vt32_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            32,
            true,
        )
        .unwrap();

        let mut run_recurrent = || {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut recurrent_state, 0);
            recurrent_prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut recurrent_q_norm,
                &mut recurrent_k_norm,
                &mut recurrent_beta,
                &mut recurrent_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            recurrent_prefill_k.encode(
                &recurrent_q_norm,
                &recurrent_k_norm,
                &recurrent_beta,
                &recurrent_decay,
                in_proj_array.allocation(),
                &mut recurrent_state,
                &mut recurrent_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                num_dv_groups as u32,
                &mut encoder,
            );
            norm_k.encode(
                &mut recurrent_out,
                in_proj_array.allocation(),
                norm_weight_array.allocation(),
                num_v_heads as u32,
                head_v_dim as u32,
                value_dim as u32,
                conv_dim as u32,
                total_proj_dim as u32,
                1e-6f32,
                suffix_len as u32,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        };

        let mut run_fused_vt32 = || {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut fused_state, 0);
            prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut q_norm_f,
                &mut k_norm_f,
                &mut beta_f,
                &mut log_decay_f,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            cumsum_k.encode(
                &log_decay_f,
                &mut g_f,
                num_v_heads as u32,
                suffix_len as u32,
                chunk_size as u32,
                &mut encoder,
            );
            gram_k.encode(
                &q_norm_f,
                &k_norm_f,
                &g_f,
                &mut kk_f,
                &mut qk_scaled_f,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            solve_k.encode(
                &kk_f,
                &beta_f,
                &g_f,
                &mut a_packed_f,
                &mut a_inv_f,
                num_v_heads as u32,
                num_k_heads as u32,
                suffix_len as u32,
                &mut encoder,
            );
            build_wu_k.encode(
                &k_norm_f,
                in_proj_array.allocation(),
                &beta_f,
                &g_f,
                &a_packed_f,
                &a_inv_f,
                &mut w_f,
                &mut u_f,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            fused_vt32_k.encode(
                &w_f,
                &u_f,
                &q_norm_f,
                &k_norm_f,
                &qk_scaled_f,
                &g_f,
                &mut fused_state,
                &mut fused_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            norm_k.encode(
                &mut fused_out,
                in_proj_array.allocation(),
                norm_weight_array.allocation(),
                num_v_heads as u32,
                head_v_dim as u32,
                value_dim as u32,
                conv_dim as u32,
                total_proj_dim as u32,
                1e-6f32,
                suffix_len as u32,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        };

        let mut run_mode_l_vt32 = || {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut mode_l_state, 0);
            prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut q_norm_m,
                &mut k_norm_m,
                &mut beta_m,
                &mut log_decay_m,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            cumsum_k.encode(
                &log_decay_m,
                &mut g_m,
                num_v_heads as u32,
                suffix_len as u32,
                chunk_size as u32,
                &mut encoder,
            );
            gram_k.encode(
                &q_norm_m,
                &k_norm_m,
                &g_m,
                &mut kk_m,
                &mut qk_scaled_m,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            solve_k.encode(
                &kk_m,
                &beta_m,
                &g_m,
                &mut a_packed_m,
                &mut a_inv_m,
                num_v_heads as u32,
                num_k_heads as u32,
                suffix_len as u32,
                &mut encoder,
            );
            solve_t_k.encode(&a_packed_m, &a_inv_m, &mut t_mat_f, num_v_heads as u32, suffix_len as u32, &mut encoder);
            mega_vt32_k.encode(
                &q_norm_m,
                &k_norm_m,
                in_proj_array.allocation(),
                &qk_scaled_m,
                &t_mat_f,
                &g_m,
                &beta_m,
                &mut mode_l_state,
                &mut mode_l_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            norm_k.encode(
                &mut mode_l_out,
                in_proj_array.allocation(),
                norm_weight_array.allocation(),
                num_v_heads as u32,
                head_v_dim as u32,
                value_dim as u32,
                conv_dim as u32,
                total_proj_dim as u32,
                1e-6f32,
                suffix_len as u32,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        };

        let warmup = |f: &mut dyn FnMut()| {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(500) {
                (f)();
            }
        };

        let iterations = if suffix_len >= 1024 {
            30
        } else {
            60
        };

        let print_row = |path: &str, result: &test_runner::perf::PerfResult| {
            eprintln!(
                "MODELBENCH\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                suffix_len, path, result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms
            );
        };

        eprintln!("\n--- T={suffix_len} (num_chunks={num_chunks}, iterations={iterations}) ---");

        warmup(&mut run_recurrent);
        let recurrent_result = run_perf_with_warmup("recurrent", 3, iterations, &mut run_recurrent);
        print_row("recurrent", &recurrent_result);

        warmup(&mut run_fused_vt32);
        let fused32_result = run_perf_with_warmup("fused_vt32", 3, iterations, &mut run_fused_vt32);
        print_row("fused_vt32", &fused32_result);

        warmup(&mut run_mode_l_vt32);
        let mode_l_result = run_perf_with_warmup("mode_l_vt32", 3, iterations, &mut run_mode_l_vt32);
        print_row("mode_l_vt32", &mode_l_result);

        eprintln!(
            "MODELVERDICT\tT={}\tmode_l={:.3}\tvs_recurrent={:.3}x\tvs_fused7={:.3}x",
            suffix_len,
            mode_l_result.median_ms,
            recurrent_result.median_ms / mode_l_result.median_ms,
            fused32_result.median_ms / mode_l_result.median_ms
        );
    }
}

// FLEET-SAFE end-to-end prefill bench for OLD-family Apple Silicon (M1, M2,
// M2-Pro, M3-Max, M4, M4-Max, M4-Pro) which have NO MXU (`mpp::tensor_ops`).
// Times ONLY the three MXU-free paths at qwen3.5 shapes:
//   recurrent               : DeltaNetPrefill path (prep -> prefill -> normGate)
//   mode_l_vt32_simd (NEW)   : Mode-L megaApply VT=32, USE_MXU=false (simdgroup)
//   mode_l_vt16_simd         : Mode-L megaApply VT=16, USE_MXU=false (simdgroup)
// CRITICAL: this bench NEVER constructs a USE_MXU=true kernel (never calls
// `.new()` on the MXU gram / solveT / megaApply variants). On MXU-less hardware
// creating such a PSO would fail, so nothing here may touch the MXU path. Every
// kernel below is constructed with the simdgroup / MXU-free variant:
//   gram USE_MXU=false, solveT USE_MXU=false, megaApply USE_MXU=false (both VTs),
//   solve RECOMPUTE_G=false (not an MXU flag). Each path warmed >=500 ms.
#[uzu_test]
#[ignore]
fn bench_delta_net_fleet_simd_vs_recurrent() {
    use std::time::{Duration, Instant};

    use test_runner::perf::run_perf_with_warmup;

    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;
    let num_dv_groups = head_v_dim.div_ceil(16);
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    eprintln!("\n=== DeltaNet FLEET simd Mode L vs Recurrent (end-to-end prefill, NO MXU) ===");
    eprintln!("  C={chunk_size} HV={num_v_heads} HK={num_k_heads} K={head_k_dim} V={head_v_dim} dtype=bf16");
    eprintln!("  scope: precompute + apply + NormGate for every path; USE_MXU=false everywhere");
    eprintln!("FLEETBENCH\tT\tpath\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");

    // (T, recurrent_ms, vt32_simd_ms, vt16_simd_ms) collected for the final table.
    let mut summary: Vec<(usize, f64, f64, f64)> = Vec::new();

    for suffix_len in [256usize, 512, 1024, 4096, 32768] {
        let num_chunks = suffix_len.div_ceil(chunk_size);

        let in_proj: Vec<bf16> =
            (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
        let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
        let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

        let context = <Metal as Backend>::Context::new().expect("context");
        let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
        let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight);
        let alloc = |data_type, len| context.create_array_zeros(&[len], data_type).into_allocation();

        // Recurrent buffers.
        let mut rec_out = alloc(DataType::BF16, suffix_len * value_dim);
        let mut rec_q = alloc(DataType::F32, suffix_len * key_dim);
        let mut rec_k = alloc(DataType::F32, suffix_len * key_dim);
        let mut rec_beta = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut rec_decay = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut rec_state = alloc(DataType::F32, state_size);

        // Shared Mode L buffers (one set; the two scan closures live in disjoint
        // scopes so their &mut borrows never overlap). T=32768 => 512 chunks; the
        // buffer sizing below is derived from num_chunks so it scales dynamically.
        let mut q_m = alloc(DataType::F32, suffix_len * key_dim);
        let mut k_m = alloc(DataType::F32, suffix_len * key_dim);
        let mut beta_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut log_decay_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut g_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut kk_m = alloc(DataType::F32, num_chunks * num_k_heads * chunk_size * chunk_size);
        let mut qk_m = alloc(DataType::F32, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut a_packed_m =
            alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size);
        let mut a_inv_m = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * block_size * block_size);
        let mut t_mat_m = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut mode_l_state = alloc(DataType::F32, state_size);
        let mut mode_l_out = alloc(DataType::BF16, suffix_len * value_dim);

        // Kernels — ALL MXU-free. There is deliberately no gram_mxu / solve_t_mxu
        // / mega_vt32 (MXU) construction anywhere in this function.
        let recurrent_prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let recurrent_prefill_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let norm_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::BF16).unwrap();
        let prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
        // solve: third arg is RECOMPUTE_G (NOT an MXU flag); false = read the
        // precomputed g buffer (the F4 lesson). MXU-free on all hardware.
        let solve_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
        // gram USE_MXU=false (simdgroup).
        let gram_simd =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, false).unwrap();
        // solveT USE_MXU=false (simdgroup).
        let solve_t_simd =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32, false)
                .unwrap();
        // megaApply VT=16, USE_MXU=false (simdgroup).
        let mega_vt16 = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            16,
            false,
        )
        .unwrap();
        // megaApply VT=32, USE_MXU=false (simdgroup) — the NEW M1-M4 candidate.
        let mega_vt32_simd = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            32,
            false,
        )
        .unwrap();

        macro_rules! encode_mode_l {
            ($mega:expr) => {{
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encoder.encode_fill(&mut mode_l_state, 0);
                prep_k.encode(
                    in_proj_array.allocation(),
                    a_log_array.allocation(),
                    dt_bias_array.allocation(),
                    &mut q_m,
                    &mut k_m,
                    &mut beta_m,
                    &mut log_decay_m,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                cumsum_k.encode(
                    &log_decay_m,
                    &mut g_m,
                    num_v_heads as u32,
                    suffix_len as u32,
                    chunk_size as u32,
                    &mut encoder,
                );
                gram_simd.encode(
                    &q_m,
                    &k_m,
                    &g_m,
                    &mut kk_m,
                    &mut qk_m,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                solve_k.encode(
                    &kk_m,
                    &beta_m,
                    &g_m,
                    &mut a_packed_m,
                    &mut a_inv_m,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                solve_t_simd.encode(
                    &a_packed_m,
                    &a_inv_m,
                    &mut t_mat_m,
                    num_v_heads as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                $mega.encode(
                    &q_m,
                    &k_m,
                    in_proj_array.allocation(),
                    &qk_m,
                    &t_mat_m,
                    &g_m,
                    &beta_m,
                    &mut mode_l_state,
                    &mut mode_l_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                norm_k.encode(
                    &mut mode_l_out,
                    in_proj_array.allocation(),
                    norm_weight_array.allocation(),
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            }};
        }

        let warmup = |f: &mut dyn FnMut()| {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(500) {
                (f)();
            }
        };
        let iterations = if suffix_len >= 1024 {
            30
        } else {
            60
        };
        let print_row = |path: &str, r: &test_runner::perf::PerfResult| {
            eprintln!(
                "FLEETBENCH\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                suffix_len, path, r.median_ms, r.min_ms, r.max_ms, r.std_dev_ms
            );
        };

        eprintln!("\n--- T={suffix_len} (num_chunks={num_chunks}, iterations={iterations}) ---");

        let mut run_recurrent = || {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut rec_state, 0);
            recurrent_prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut rec_q,
                &mut rec_k,
                &mut rec_beta,
                &mut rec_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            recurrent_prefill_k.encode(
                &rec_q,
                &rec_k,
                &rec_beta,
                &rec_decay,
                in_proj_array.allocation(),
                &mut rec_state,
                &mut rec_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                num_dv_groups as u32,
                &mut encoder,
            );
            norm_k.encode(
                &mut rec_out,
                in_proj_array.allocation(),
                norm_weight_array.allocation(),
                num_v_heads as u32,
                head_v_dim as u32,
                value_dim as u32,
                conv_dim as u32,
                total_proj_dim as u32,
                1e-6f32,
                suffix_len as u32,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        };
        warmup(&mut run_recurrent);
        let rec_res = run_perf_with_warmup("recurrent", 3, iterations, &mut run_recurrent);
        print_row("recurrent", &rec_res);
        drop(run_recurrent);

        let vt32_simd_res = {
            let mut run_vt32_simd = || encode_mode_l!(mega_vt32_simd);
            warmup(&mut run_vt32_simd);
            run_perf_with_warmup("mode_l_vt32_simd", 3, iterations, &mut run_vt32_simd)
        };
        print_row("mode_l_vt32_simd", &vt32_simd_res);

        let vt16_simd_res = {
            let mut run_vt16_simd = || encode_mode_l!(mega_vt16);
            warmup(&mut run_vt16_simd);
            run_perf_with_warmup("mode_l_vt16_simd", 3, iterations, &mut run_vt16_simd)
        };
        print_row("mode_l_vt16_simd", &vt16_simd_res);

        eprintln!(
            "FLEETVERDICT\tT={}\trec={:.3}\tvt32_simd={:.3}\tvt16_simd={:.3}\t\
             vt32s_vs_rec={:.3}x\tvt16s_vs_rec={:.3}x\tvt32s_vs_vt16s={:.3}x",
            suffix_len,
            rec_res.median_ms,
            vt32_simd_res.median_ms,
            vt16_simd_res.median_ms,
            rec_res.median_ms / vt32_simd_res.median_ms,
            rec_res.median_ms / vt16_simd_res.median_ms,
            vt16_simd_res.median_ms / vt32_simd_res.median_ms,
        );

        summary.push((suffix_len, rec_res.median_ms, vt32_simd_res.median_ms, vt16_simd_res.median_ms));
    }

    // Final clean table across all T.
    eprintln!("\n=== FLEET simd Mode L vs Recurrent summary (median ms; NO MXU) ===");
    eprintln!(
        "{:>7}  {:>11}  {:>11}  {:>11}  {:>12}  {:>12}",
        "T", "recurrent", "vt32-simd", "vt16-simd", "vt32/rec", "vt16/rec"
    );
    for (t, rec, vt32, vt16) in &summary {
        eprintln!(
            "{:>7}  {:>11.3}  {:>11.3}  {:>11.3}  {:>11.3}x  {:>11.3}x",
            t,
            rec,
            vt32,
            vt16,
            rec / vt32,
            rec / vt16
        );
    }
}

// End-to-end Mode L sweep comparing the recurrent baseline against the three MXU
// configurations of the mega pipeline (prep, cumsum, gram, solve, solveT,
// megaApply, normGate):
//   (A) all-MXU:       gram(MXU) + solveT(MXU) + megaApply VT=32 (MXU)
//   (B) all-simdgroup: gram(simd) + solveT(simd) + megaApply VT=16 (simd)  [M1-M4 proxy]
//   (vt32_simd)        gram(simd) + solveT(simd) + megaApply VT=32 (simd)  [NEW M1-M4 candidate]
//   (mixed)            gram(simd) + solveT(simd) + megaApply VT=32 (MXU)   [current reference]
// The SCAN4WAY line isolates the scan variant (matched simd precompute):
// recurrent / VT16-simd (B) / VT32-simd (NEW) / VT32-MXU (mixed).
// Each path warmed >=500 ms before timing. At T=4096 also emits a per-kernel
// isolated median for gram / solveT / buildWU / megaApply, config A vs B, to
// attribute how much each precompute kernel gained from MXU.
#[uzu_test]
#[ignore]
fn bench_delta_net_mode_l_mxu_configs() {
    use std::time::{Duration, Instant};

    use test_runner::perf::run_perf_with_warmup;

    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;
    let bv = 32usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;
    let num_dv_groups = head_v_dim.div_ceil(16);
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    eprintln!("\n=== DeltaNet Mode L MXU configs (end-to-end prefill) ===");
    eprintln!("  scope: precompute + apply + NormGate for every path");
    eprintln!(
        "  A=all-MXU(gram+solveT MXU, mega VT32)  B=all-simd(gram+solveT simd, mega VT16)  mixed=simd precompute + mega VT32"
    );
    eprintln!("MXUBENCH\tT\tpath\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");

    for suffix_len in [128usize, 256, 512, 1024, 2048, 4096, 8192, 32768] {
        let num_chunks = suffix_len.div_ceil(chunk_size);

        let in_proj: Vec<bf16> =
            (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
        let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
        let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

        let context = <Metal as Backend>::Context::new().expect("context");
        let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
        let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight);
        let alloc = |data_type, len| context.create_array_zeros(&[len], data_type).into_allocation();

        // Recurrent buffers.
        let mut rec_out = alloc(DataType::BF16, suffix_len * value_dim);
        let mut rec_q = alloc(DataType::F32, suffix_len * key_dim);
        let mut rec_k = alloc(DataType::F32, suffix_len * key_dim);
        let mut rec_beta = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut rec_decay = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut rec_state = alloc(DataType::F32, state_size);

        // Shared Mode L buffers (one set; the three config closures live in
        // disjoint scopes so their &mut borrows never overlap).
        let mut q_m = alloc(DataType::F32, suffix_len * key_dim);
        let mut k_m = alloc(DataType::F32, suffix_len * key_dim);
        let mut beta_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut log_decay_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut g_m = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut kk_m = alloc(DataType::F32, num_chunks * num_k_heads * chunk_size * chunk_size);
        let mut qk_m = alloc(DataType::F32, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut a_packed_m =
            alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size);
        let mut a_inv_m = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * block_size * block_size);
        let mut t_mat_m = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut mode_l_state = alloc(DataType::F32, state_size);
        let mut mode_l_out = alloc(DataType::BF16, suffix_len * value_dim);

        // Kernels.
        let recurrent_prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let recurrent_prefill_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let norm_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::BF16).unwrap();
        let prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
        let solve_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
        let gram_simd =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, false).unwrap();
        let gram_mxu =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, true).unwrap();
        let solve_t_simd =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32, false)
                .unwrap();
        let solve_t_mxu =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32, true).unwrap();
        let mega_vt16 = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            16,
            false,
        )
        .unwrap();
        // NEW decoupled variant: VT=32 on the simdgroup backend (M1-M4 candidate).
        let mega_vt32_simd = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            32,
            false,
        )
        .unwrap();
        let mega_vt32 = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            32,
            true,
        )
        .unwrap();

        macro_rules! encode_mode_l {
            ($gram:expr, $solve_t:expr, $mega:expr) => {{
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encoder.encode_fill(&mut mode_l_state, 0);
                prep_k.encode(
                    in_proj_array.allocation(),
                    a_log_array.allocation(),
                    dt_bias_array.allocation(),
                    &mut q_m,
                    &mut k_m,
                    &mut beta_m,
                    &mut log_decay_m,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                cumsum_k.encode(
                    &log_decay_m,
                    &mut g_m,
                    num_v_heads as u32,
                    suffix_len as u32,
                    chunk_size as u32,
                    &mut encoder,
                );
                $gram.encode(
                    &q_m,
                    &k_m,
                    &g_m,
                    &mut kk_m,
                    &mut qk_m,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                solve_k.encode(
                    &kk_m,
                    &beta_m,
                    &g_m,
                    &mut a_packed_m,
                    &mut a_inv_m,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                $solve_t.encode(
                    &a_packed_m,
                    &a_inv_m,
                    &mut t_mat_m,
                    num_v_heads as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                $mega.encode(
                    &q_m,
                    &k_m,
                    in_proj_array.allocation(),
                    &qk_m,
                    &t_mat_m,
                    &g_m,
                    &beta_m,
                    &mut mode_l_state,
                    &mut mode_l_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                norm_k.encode(
                    &mut mode_l_out,
                    in_proj_array.allocation(),
                    norm_weight_array.allocation(),
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            }};
        }

        let warmup = |f: &mut dyn FnMut()| {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(500) {
                (f)();
            }
        };
        let iterations = if suffix_len >= 1024 {
            30
        } else {
            60
        };
        let print_row = |path: &str, r: &test_runner::perf::PerfResult| {
            eprintln!(
                "MXUBENCH\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                suffix_len, path, r.median_ms, r.min_ms, r.max_ms, r.std_dev_ms
            );
        };

        eprintln!("\n--- T={suffix_len} (num_chunks={num_chunks}, iterations={iterations}) ---");

        let mut run_recurrent = || {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut rec_state, 0);
            recurrent_prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut rec_q,
                &mut rec_k,
                &mut rec_beta,
                &mut rec_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            recurrent_prefill_k.encode(
                &rec_q,
                &rec_k,
                &rec_beta,
                &rec_decay,
                in_proj_array.allocation(),
                &mut rec_state,
                &mut rec_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                num_dv_groups as u32,
                &mut encoder,
            );
            norm_k.encode(
                &mut rec_out,
                in_proj_array.allocation(),
                norm_weight_array.allocation(),
                num_v_heads as u32,
                head_v_dim as u32,
                value_dim as u32,
                conv_dim as u32,
                total_proj_dim as u32,
                1e-6f32,
                suffix_len as u32,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        };
        warmup(&mut run_recurrent);
        let rec_res = run_perf_with_warmup("recurrent", 3, iterations, &mut run_recurrent);
        print_row("recurrent", &rec_res);
        drop(run_recurrent);

        let b_res = {
            let mut run_b = || encode_mode_l!(gram_simd, solve_t_simd, mega_vt16);
            warmup(&mut run_b);
            run_perf_with_warmup("B_all_simd", 3, iterations, &mut run_b)
        };
        print_row("B_all_simd", &b_res);

        // NEW: VT=32 simdgroup scan on the same simd precompute as B/mixed, so
        // the recurrent / VT16-simd / VT32-simd / VT32-MXU(mixed) rows isolate the
        // scan variant. This is the M1-M4 candidate row.
        let vt32_simd_res = {
            let mut run_vt32_simd = || encode_mode_l!(gram_simd, solve_t_simd, mega_vt32_simd);
            warmup(&mut run_vt32_simd);
            run_perf_with_warmup("vt32_simd", 3, iterations, &mut run_vt32_simd)
        };
        print_row("vt32_simd", &vt32_simd_res);

        let a_res = {
            let mut run_a = || encode_mode_l!(gram_mxu, solve_t_mxu, mega_vt32);
            warmup(&mut run_a);
            run_perf_with_warmup("A_all_mxu", 3, iterations, &mut run_a)
        };
        print_row("A_all_mxu", &a_res);

        let mixed_res = {
            let mut run_mixed = || encode_mode_l!(gram_simd, solve_t_simd, mega_vt32);
            warmup(&mut run_mixed);
            run_perf_with_warmup("mixed", 3, iterations, &mut run_mixed)
        };
        print_row("mixed_ref", &mixed_res);

        // Four-way scan comparison (matched simd precompute): recurrent /
        // VT16-simd (B) / VT32-simd (NEW) / VT32-MXU (mixed), with speedups vs
        // recurrent for each scan variant.
        eprintln!(
            "SCAN4WAY\tT={}\trec={:.3}\tvt16_simd={:.3}\tvt32_simd={:.3}\tvt32_mxu={:.3}\t\
             vt16s_vs_rec={:.3}x\tvt32s_vs_rec={:.3}x\tvt32m_vs_rec={:.3}x",
            suffix_len,
            rec_res.median_ms,
            b_res.median_ms,
            vt32_simd_res.median_ms,
            mixed_res.median_ms,
            rec_res.median_ms / b_res.median_ms,
            rec_res.median_ms / vt32_simd_res.median_ms,
            rec_res.median_ms / mixed_res.median_ms
        );
        eprintln!(
            "MXUVERDICT\tT={}\tA={:.3}\tB={:.3}\tmixed={:.3}\trec={:.3}\tA_vs_rec={:.3}x\tA_vs_B={:.3}x\tA_vs_mixed={:.3}x",
            suffix_len,
            a_res.median_ms,
            b_res.median_ms,
            mixed_res.median_ms,
            rec_res.median_ms,
            rec_res.median_ms / a_res.median_ms,
            b_res.median_ms / a_res.median_ms,
            mixed_res.median_ms / a_res.median_ms
        );

        // Per-kernel isolated A-vs-B breakdown at T=4096.
        if suffix_len == 4096 {
            let mut w_m = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * head_k_dim);
            let mut u_m = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * head_v_dim);
            let build_wu_simd = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedBuildWUKernel::new(
                &context,
                DataType::BF16,
                DataType::BF16,
                128,
                64,
                bv as u32,
                false,
                false,
            )
            .unwrap();
            let build_wu_mxu = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedBuildWUKernel::new(
                &context,
                DataType::BF16,
                DataType::BF16,
                128,
                64,
                bv as u32,
                false,
                true,
            )
            .unwrap();

            // Populate all precompute inputs once (simd precompute is fine as the
            // shared input state for the isolated timings below).
            encode_mode_l!(gram_simd, solve_t_simd, mega_vt32);

            eprintln!("MXUBRK\tT\tkernel\tconfig\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");
            let print_brk = |kernel: &str, cfg: &str, r: &test_runner::perf::PerfResult| {
                eprintln!(
                    "MXUBRK\t4096\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    kernel, cfg, r.median_ms, r.min_ms, r.max_ms, r.std_dev_ms
                );
            };

            macro_rules! isolate {
                ($body:expr) => {{
                    let mut run = || {
                        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                        $body(&mut encoder);
                        encoder.end_encoding().submit().wait_until_completed().unwrap();
                    };
                    warmup(&mut run);
                    run_perf_with_warmup("iso", 3, 40, &mut run)
                }};
            }

            let gram_a = isolate!(|enc: &mut _| gram_mxu.encode(
                &q_m,
                &k_m,
                &g_m,
                &mut kk_m,
                &mut qk_m,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                enc
            ));
            let gram_b = isolate!(|enc: &mut _| gram_simd.encode(
                &q_m,
                &k_m,
                &g_m,
                &mut kk_m,
                &mut qk_m,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                enc
            ));
            print_brk("gram", "A_mxu", &gram_a);
            print_brk("gram", "B_simd", &gram_b);

            let solvet_a = isolate!(|enc: &mut _| solve_t_mxu.encode(
                &a_packed_m,
                &a_inv_m,
                &mut t_mat_m,
                num_v_heads as u32,
                suffix_len as u32,
                enc
            ));
            let solvet_b = isolate!(|enc: &mut _| solve_t_simd.encode(
                &a_packed_m,
                &a_inv_m,
                &mut t_mat_m,
                num_v_heads as u32,
                suffix_len as u32,
                enc
            ));
            print_brk("solveT", "A_mxu", &solvet_a);
            print_brk("solveT", "B_simd", &solvet_b);

            let buildwu_a = isolate!(|enc: &mut _| build_wu_mxu.encode(
                &k_m,
                in_proj_array.allocation(),
                &beta_m,
                &g_m,
                &a_packed_m,
                &a_inv_m,
                &mut w_m,
                &mut u_m,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                enc
            ));
            let buildwu_b = isolate!(|enc: &mut _| build_wu_simd.encode(
                &k_m,
                in_proj_array.allocation(),
                &beta_m,
                &g_m,
                &a_packed_m,
                &a_inv_m,
                &mut w_m,
                &mut u_m,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                enc
            ));
            print_brk("buildWU", "A_mxu", &buildwu_a);
            print_brk("buildWU", "B_simd", &buildwu_b);

            let mega_a = isolate!(|enc: &mut _| mega_vt32.encode(
                &q_m,
                &k_m,
                in_proj_array.allocation(),
                &qk_m,
                &t_mat_m,
                &g_m,
                &beta_m,
                &mut mode_l_state,
                &mut mode_l_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                enc
            ));
            let mega_b = isolate!(|enc: &mut _| mega_vt16.encode(
                &q_m,
                &k_m,
                in_proj_array.allocation(),
                &qk_m,
                &t_mat_m,
                &g_m,
                &beta_m,
                &mut mode_l_state,
                &mut mode_l_out,
                num_v_heads as u32,
                num_k_heads as u32,
                head_v_dim as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                enc
            ));
            print_brk("megaApply", "A_mxu", &mega_a);
            print_brk("megaApply", "B_simd", &mega_b);
        }
    }
}

// Per-kernel wall-clock breakdown of the Mode L pipeline at T in {256, 4096}.
// Times each kernel isolated (>=500 ms warmup) plus grouped precompute and the
// full pipeline single-encoder, so the in-context apply is precompute-vs-apply
// attributable exactly like the fused breakdown.
#[uzu_test]
#[ignore]
fn bench_delta_net_mode_l_breakdown() {
    use std::time::{Duration, Instant};

    use test_runner::perf::run_perf_with_warmup;

    let num_v_heads = 48usize;
    let num_k_heads = 16usize;
    let head_k_dim = 128usize;
    let head_v_dim = 128usize;
    let chunk_size = 64usize;

    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_v_dim * head_k_dim;

    eprintln!("\n=== DeltaNet Mode L pipeline per-kernel breakdown ===");
    eprintln!("MODELBRK\tT\tkernel\tstage\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");

    for suffix_len in [256usize, 4096] {
        let num_chunks = suffix_len.div_ceil(chunk_size);
        eprintln!("\n--- Mode L per-kernel breakdown T={suffix_len} num_chunks={num_chunks} ---");

        let in_proj: Vec<bf16> =
            (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
        let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
        let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

        let context = <Metal as Backend>::Context::new().expect("context");
        let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj);
        let a_log_array = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight);

        let alloc = |data_type, len| context.create_array_zeros(&[len], data_type).into_allocation();

        let mut q_norm = alloc(DataType::F32, suffix_len * key_dim);
        let mut k_norm = alloc(DataType::F32, suffix_len * key_dim);
        let mut beta = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut log_decay = alloc(DataType::F32, suffix_len * num_v_heads);
        let mut g = alloc(DataType::F32, suffix_len * num_v_heads);
        let block_size = 16usize;
        let num_blocks = chunk_size.div_ceil(block_size);
        let num_col_pairs = num_blocks.div_ceil(2);
        let mut kk = alloc(DataType::F32, num_chunks * num_k_heads * chunk_size * chunk_size);
        let mut qk_scaled = alloc(DataType::F32, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut a_packed =
            alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size);
        let mut a_inv = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * block_size * block_size);
        let mut t_mat = alloc(DataType::BF16, num_chunks * num_v_heads * chunk_size * chunk_size);
        let mut mega_state = alloc(DataType::F32, state_size);
        let mut mega_out = alloc(DataType::BF16, suffix_len * value_dim);

        let prep_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
                .unwrap();
        let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
        let gram_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64, false).unwrap();
        let solve_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
        let solve_t_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32, false)
                .unwrap();
        let norm_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::BF16).unwrap();
        let mega_vt32_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            32,
            true,
        )
        .unwrap();

        // Prime buffers once.
        {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            encoder.encode_fill(&mut mega_state, 0);
            prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut q_norm,
                &mut k_norm,
                &mut beta,
                &mut log_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, &mut encoder);
            gram_k.encode(
                &q_norm,
                &k_norm,
                &g,
                &mut kk,
                &mut qk_scaled,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
            solve_k.encode(
                &kk,
                &beta,
                &g,
                &mut a_packed,
                &mut a_inv,
                num_v_heads as u32,
                num_k_heads as u32,
                suffix_len as u32,
                &mut encoder,
            );
            solve_t_k.encode(&a_packed, &a_inv, &mut t_mat, num_v_heads as u32, suffix_len as u32, &mut encoder);
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        }

        let iterations = 50usize;
        let warmup = |f: &mut dyn FnMut()| {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(500) {
                f();
            }
        };

        let mut results: Vec<(&str, test_runner::perf::PerfResult)> = Vec::new();

        macro_rules! time_kernel {
            ($name:expr, |$enc:ident| $body:block) => {{
                let mut run = || {
                    let mut $enc = Encoder::new(context.as_ref()).expect("encoder");
                    $body
                    $enc.end_encoding().submit().wait_until_completed().unwrap();
                };
                warmup(&mut run);
                let result = run_perf_with_warmup($name, 3, iterations, &mut run);
                eprintln!(
                    "MODELBRK\t{}\t{}\tprecompute\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    suffix_len, $name, result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms
                );
                results.push(($name, result));
            }};
        }

        time_kernel!("prep", |encoder| {
            prep_k.encode(
                in_proj_array.allocation(),
                a_log_array.allocation(),
                dt_bias_array.allocation(),
                &mut q_norm,
                &mut k_norm,
                &mut beta,
                &mut log_decay,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                value_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        time_kernel!("cumsum", |encoder| {
            cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, &mut encoder);
        });
        time_kernel!("gram", |encoder| {
            gram_k.encode(
                &q_norm,
                &k_norm,
                &g,
                &mut kk,
                &mut qk_scaled,
                num_v_heads as u32,
                num_k_heads as u32,
                key_dim as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        time_kernel!("solve", |encoder| {
            solve_k.encode(
                &kk,
                &beta,
                &g,
                &mut a_packed,
                &mut a_inv,
                num_v_heads as u32,
                num_k_heads as u32,
                suffix_len as u32,
                &mut encoder,
            );
        });
        time_kernel!("solve_t", |encoder| {
            solve_t_k.encode(&a_packed, &a_inv, &mut t_mat, num_v_heads as u32, suffix_len as u32, &mut encoder);
        });

        // The mega apply dispatch (its own stage label).
        {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                mega_vt32_k.encode(
                    &q_norm,
                    &k_norm,
                    in_proj_array.allocation(),
                    &qk_scaled,
                    &t_mat,
                    &g,
                    &beta,
                    &mut mega_state,
                    &mut mega_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            let result = run_perf_with_warmup("mega_apply_vt32", 3, iterations, &mut run);
            eprintln!(
                "MODELBRK\t{}\t{}\tapply\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                suffix_len, "mega_apply_vt32", result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms
            );
            results.push(("mega_apply_vt32", result));
        }

        let norm_result = {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                norm_k.encode(
                    &mut mega_out,
                    in_proj_array.allocation(),
                    norm_weight_array.allocation(),
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            run_perf_with_warmup("norm_gate", 3, iterations, &mut run)
        };

        macro_rules! encode_precompute {
            ($encoder:expr) => {{
                prep_k.encode(
                    in_proj_array.allocation(),
                    a_log_array.allocation(),
                    dt_bias_array.allocation(),
                    &mut q_norm,
                    &mut k_norm,
                    &mut beta,
                    &mut log_decay,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    $encoder,
                );
                cumsum_k.encode(&log_decay, &mut g, num_v_heads as u32, suffix_len as u32, chunk_size as u32, $encoder);
                gram_k.encode(
                    &q_norm,
                    &k_norm,
                    &g,
                    &mut kk,
                    &mut qk_scaled,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    suffix_len as u32,
                    $encoder,
                );
                solve_k.encode(
                    &kk,
                    &beta,
                    &g,
                    &mut a_packed,
                    &mut a_inv,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    suffix_len as u32,
                    $encoder,
                );
                solve_t_k.encode(&a_packed, &a_inv, &mut t_mat, num_v_heads as u32, suffix_len as u32, $encoder);
            }};
        }

        let precompute_group = {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encode_precompute!(&mut encoder);
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            run_perf_with_warmup("precompute_chain", 3, iterations, &mut run)
        };

        let pipeline_group = {
            let mut run = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encoder.encode_fill(&mut mega_state, 0);
                encode_precompute!(&mut encoder);
                mega_vt32_k.encode(
                    &q_norm,
                    &k_norm,
                    in_proj_array.allocation(),
                    &qk_scaled,
                    &t_mat,
                    &g,
                    &beta,
                    &mut mega_state,
                    &mut mega_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                norm_k.encode(
                    &mut mega_out,
                    in_proj_array.allocation(),
                    norm_weight_array.allocation(),
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run);
            run_perf_with_warmup("full_pipeline", 3, iterations, &mut run)
        };

        let apply_ms = results.iter().find(|(n, _)| *n == "mega_apply_vt32").unwrap().1.median_ms;
        let precompute_ms: f64 =
            results.iter().filter(|(n, _)| *n != "mega_apply_vt32").map(|(_, r)| r.median_ms).sum();
        let e2e = pipeline_group.median_ms;
        let precompute_ctx = precompute_group.median_ms;
        let apply_ctx = (e2e - precompute_ctx - norm_result.median_ms).max(0.0);

        eprintln!("\n--- Mode L pipeline breakdown (T={suffix_len}) ---");
        eprintln!("kernel               median_ms");
        let mut sorted: Vec<&(&str, test_runner::perf::PerfResult)> = results.iter().collect();
        sorted.sort_by(|a, b| b.1.median_ms.partial_cmp(&a.1.median_ms).unwrap());
        for (name, r) in sorted {
            eprintln!("  {:<18} {:8.4}", name, r.median_ms);
        }
        eprintln!("  {:<18} {:8.4}", "(norm_gate)", norm_result.median_ms);
        eprintln!("SUMMARY(isolated)\tprecompute={:.4}ms\tapply={:.4}ms", precompute_ms, apply_ms);
        eprintln!(
            "SUMMARY(grouped)\te2e={:.4}ms\tprecompute_chain={:.4}ms\tapply_in_context={:.4}ms\tnorm={:.4}ms",
            e2e, precompute_ctx, apply_ctx, norm_result.median_ms
        );
    }
}
