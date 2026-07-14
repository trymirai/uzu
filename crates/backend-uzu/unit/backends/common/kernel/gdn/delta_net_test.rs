#![cfg(metal_backend)]

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{
                Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
                DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
            },
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_prefix_to_vec, allocation_to_vec},
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

    let w_array = alloc_allocation_with_data::<B, f32>(&context, w);
    let b_array = alloc_allocation_with_data::<B, f32>(&context, b);
    let mut in_out = alloc_allocation_with_data::<B, f32>(&context, in_proj);
    let mut state_allocation = alloc_allocation_with_data::<B, f32>(&context, state);

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(&context, DataType::F32, true)
        .expect("Failed to create kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &w_array,
        Some(&b_array),
        &mut in_out,
        &mut state_allocation,
        kernel_size,
        conv_dim,
        state_stride,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = allocation_prefix_to_vec::<B, f32>(&in_out, conv_dim as usize);
    let new_state = allocation_to_vec::<B, f32>(&state_allocation);
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

    let in_proj_array = alloc_allocation_with_data::<B, f32>(&context, in_proj);
    let a_log_array = alloc_allocation_with_data::<B, f32>(&context, a_log);
    let dt_bias_array = alloc_allocation_with_data::<B, f32>(&context, dt_bias);
    let norm_weight_array = alloc_allocation_with_data::<B, f32>(&context, norm_weight);
    let mut state_allocation = alloc_allocation_with_data::<B, f32>(&context, state);
    let mut out = alloc_allocation::<B, f32>(&context, value_dim as usize);

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetUpdateKernel::new(&context, DataType::F32, head_k_dim)
        .expect("Failed to create kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &in_proj_array,
        &a_log_array,
        &dt_bias_array,
        &norm_weight_array,
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

    let out = allocation_to_vec::<B, f32>(&out);
    let new_state = allocation_to_vec::<B, f32>(&state_allocation);
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
    let state_array = alloc_allocation_with_data::<Metal, f32>(&context, &init_state);
    let mut in_proj_array = alloc_allocation_with_data::<Metal, bf16>(&context, &in_proj);
    let w_array = alloc_allocation_with_data::<Metal, f32>(&context, &w);
    let b_array = alloc_allocation_with_data::<Metal, f32>(&context, &b);

    let padded_len = (tap_count + suffix_len) * total_proj_dim;
    let mut padded_array = alloc_allocation::<Metal, f32>(&context, padded_len);
    let mut state_out_array = alloc_allocation::<Metal, f32>(&context, conv_dim * tap_count);

    let pack_kernel =
        <<Metal as Backend>::Kernels as Kernels>::Conv1dPackKernel::new(&context, DataType::F32, DataType::BF16)
            .expect("pack");
    let scan_kernel =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetConvScanKernel::new(&context, DataType::BF16, true)
            .expect("scan");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    pack_kernel.encode(
        &state_array,
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
        &w_array,
        Some(&b_array),
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

    let in_proj_result: Vec<bf16> = allocation_to_vec(&in_proj_array);
    let in_proj_result: Vec<f32> = in_proj_result.into_iter().map(f32::from).collect();
    let mut scan_outputs = vec![0.0f32; suffix_len * conv_dim];
    for t in 0..suffix_len {
        scan_outputs[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&in_proj_result[t * total_proj_dim..t * total_proj_dim + conv_dim]);
    }
    let scan_state: Vec<f32> = allocation_to_vec(&state_out_array);

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
    let in_proj_array = alloc_allocation_with_data::<Metal, T>(&context, in_proj);
    let a_log_array = alloc_allocation_with_data::<Metal, f32>(&context, a_log);
    let dt_bias_array = alloc_allocation_with_data::<Metal, f32>(&context, dt_bias);
    let norm_weight_array = alloc_allocation_with_data::<Metal, f32>(&context, norm_weight);
    let mut state_array = alloc_allocation_with_data::<Metal, f32>(&context, state);
    let mut out_array = alloc_allocation::<Metal, T>(&context, suffix_len * value_dim);
    let mut q_norm_array = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut k_norm_array = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);

    let mut beta_array = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut decay_array = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        T::data_type(),
        head_k_dim as u32,
        false,
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
        &in_proj_array,
        &a_log_array,
        &dt_bias_array,
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
        &in_proj_array,
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
        &in_proj_array,
        &norm_weight_array,
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

    let out: Vec<T> = allocation_to_vec(&out_array);
    let out = out.into_iter().map(|value| value.to_f32().expect("output to f32")).collect();
    let state = allocation_to_vec(&state_array);
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
    let cpu_in_proj = alloc_allocation_with_data::<Cpu, f32>(&cpu_ctx, &in_proj);
    let cpu_a_log = alloc_allocation_with_data::<Cpu, f32>(&cpu_ctx, &a_log);
    let cpu_dt_bias = alloc_allocation_with_data::<Cpu, f32>(&cpu_ctx, &dt_bias);
    let mut cpu_q = alloc_allocation::<Cpu, f32>(&cpu_ctx, suffix_len * key_dim);
    let mut cpu_k = alloc_allocation::<Cpu, f32>(&cpu_ctx, suffix_len * key_dim);
    let mut cpu_beta = alloc_allocation::<Cpu, f32>(&cpu_ctx, suffix_len * num_v_heads);
    let mut cpu_decay = alloc_allocation::<Cpu, f32>(&cpu_ctx, suffix_len * num_v_heads);

    let cpu_prep = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &cpu_ctx,
        DataType::F32,
        head_k_dim as u32,
        false,
    )
    .unwrap();
    let mut cpu_enc = Encoder::new(cpu_ctx.as_ref()).expect("encoder");
    cpu_prep.encode(
        &cpu_in_proj,
        &cpu_a_log,
        &cpu_dt_bias,
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

    let ref_q: Vec<f32> = allocation_to_vec(&cpu_q);
    let ref_k: Vec<f32> = allocation_to_vec(&cpu_k);
    let ref_beta: Vec<f32> = allocation_to_vec(&cpu_beta);
    let ref_decay: Vec<f32> = allocation_to_vec(&cpu_decay);

    // Metal
    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = alloc_allocation_with_data::<Metal, f32>(&context, &in_proj);
    let a_log_array = alloc_allocation_with_data::<Metal, f32>(&context, &a_log);
    let dt_bias_array = alloc_allocation_with_data::<Metal, f32>(&context, &dt_bias);
    let mut q_norm_array = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut k_norm_array = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);

    let mut beta_array = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut decay_array = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::F32,
        head_k_dim as u32,
        false,
    )
    .unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        &in_proj_array,
        &a_log_array,
        &dt_bias_array,
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

    let gpu_q: Vec<f32> = allocation_to_vec(&q_norm_array);
    let gpu_k: Vec<f32> = allocation_to_vec(&k_norm_array);
    let gpu_beta: Vec<f32> = allocation_to_vec(&beta_array);
    let gpu_decay: Vec<f32> = allocation_to_vec(&decay_array);

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
    let in_proj_array = alloc_allocation_with_data::<Metal, f32>(&context, &in_proj);
    let a_log_array = alloc_allocation_with_data::<Metal, f32>(&context, &a_log);
    let dt_bias_array = alloc_allocation_with_data::<Metal, f32>(&context, &dt_bias);
    let norm_weight_array = alloc_allocation_with_data::<Metal, f32>(&context, &norm_weight);
    let mut out_array = alloc_allocation::<Metal, f32>(&context, suffix_len * value_dim);
    let mut q_norm_array = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut k_norm_array = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);

    let mut beta_array = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut decay_array = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);

    let num_dv_groups = head_v_dim.div_ceil(16) as u32;

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::F32,
        head_k_dim as u32,
        false,
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
            &in_proj_array,
            &a_log_array,
            &dt_bias_array,
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
    let mut state_array = alloc_allocation::<Metal, f32>(&context, state_size);

    let prefill_result = run_perf_with_warmup("prep+prefill+norm_gate", 5, 50, || {
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        encoder.encode_fill(&mut state_array, 0);
        prep_k.encode(
            &in_proj_array,
            &a_log_array,
            &dt_bias_array,
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
            &in_proj_array,
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
            &in_proj_array,
            &norm_weight_array,
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
