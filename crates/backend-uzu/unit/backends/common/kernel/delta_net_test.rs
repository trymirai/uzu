#![cfg(metal_backend)]

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            kernel::{
                Conv1dPackKernel, DeltaNetChunkedCumsumKernel, DeltaNetChunkedGramKernel,
                DeltaNetChunkedMegaApplyKernel, DeltaNetChunkedPrepKernel, DeltaNetChunkedSolveKernel,
                DeltaNetChunkedSolveTKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
                DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
            },
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_prefix_to_vec, allocation_to_vec},
};

fn alloc_dtype<B: Backend>(
    context: &B::Context,
    elements_count: usize,
    data_type: DataType,
) -> Allocation<B> {
    alloc_allocation::<B, u8>(context, elements_count * data_type.size_in_bytes())
}

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
    let in_proj_array = alloc_allocation_with_data::<Metal, _>(&context, in_proj);
    let a_log_array = alloc_allocation_with_data::<Metal, _>(&context, a_log);
    let dt_bias_array = alloc_allocation_with_data::<Metal, _>(&context, dt_bias);
    let mut state_array = alloc_allocation_with_data::<Metal, _>(&context, init_state);
    // The prefill/prep dtype must match the (bf16) in_proj activation dtype;
    // out is therefore bf16 (same as the existing chunked-vs-recurrent bench).
    let mut out_array = alloc_allocation::<Metal, bf16>(&context, suffix_len * value_dim);
    let mut q_norm = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut k_norm = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut beta = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut decay = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);

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
        &in_proj_array,
        &a_log_array,
        &dt_bias_array,
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
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out_bf16: Vec<bf16> = crate::tests::helpers::allocation_to_vec(&out_array);
    let out: Vec<f32> = out_bf16.iter().map(|value| value.to_f32()).collect();
    (out, crate::tests::helpers::allocation_to_vec(&state_array))
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
    let q_a = alloc_allocation_with_data::<B, _>(&context, q_norm);
    let k_a = alloc_allocation_with_data::<B, _>(&context, k_norm);
    let in_a = alloc_allocation_with_data::<B, _>(&context, in_proj);
    let qk_a = alloc_allocation_with_data::<B, _>(&context, qk_scaled);
    let t_a = alloc_allocation_with_data::<B, _>(&context, t_mat);
    let g_a = alloc_allocation_with_data::<B, _>(&context, g);
    let beta_a = alloc_allocation_with_data::<B, _>(&context, beta);
    let mut state_alloc = alloc_allocation_with_data::<B, _>(&context, state);
    let mut out_alloc = alloc_allocation::<B, f32>(&context, suffix_len * value_dim);

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
        &q_a,
        &k_a,
        &in_a,
        &qk_a,
        &t_a,
        &g_a,
        &beta_a,
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
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bf16>, Vec<f32>, Vec<f32>) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = alloc_allocation_with_data::<Metal, _>(&context, in_proj);
    let a_log_array = alloc_allocation_with_data::<Metal, _>(&context, a_log);
    let dt_bias_array = alloc_allocation_with_data::<Metal, _>(&context, dt_bias);

    let mut q_norm = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut k_norm = alloc_allocation::<Metal, f32>(&context, suffix_len * key_dim);
    let mut beta = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut log_decay = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut g = alloc_allocation::<Metal, f32>(&context, suffix_len * num_v_heads);
    let mut kk = alloc_allocation::<Metal, f32>(&context, num_chunks * num_k_heads * chunk_size * chunk_size);
    let mut qk_scaled = alloc_allocation::<Metal, f32>(&context, num_chunks * num_v_heads * chunk_size * chunk_size);
    let mut a_packed = alloc_allocation::<Metal, f32>(
        &context,
        num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size,
    );
    let mut a_inv =
        alloc_allocation::<Metal, f32>(&context, num_chunks * num_v_heads * num_blocks * block_size * block_size);
    let mut t_mat = alloc_allocation::<Metal, bf16>(&context, num_chunks * num_v_heads * chunk_size * chunk_size);

    let prep_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(&context, DataType::BF16, 128)
            .unwrap();
    let cumsum_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(&context).unwrap();
    let gram_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64).unwrap();
    let solve_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
    let solve_t_k =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32).unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        &in_proj_array,
        &a_log_array,
        &dt_bias_array,
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

    // Single precompute chain (simdgroup gram/solveT). Precompute is
    // VT-independent and backend-independent; megaApply then runs at VT=32 on
    // both the simdgroup (M1-M4) and MXU (M5) backends.
    let precompute_simd = run_mode_l_precompute_metal(
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

        // The two shipping megaApply configs, both at VT=32 over the same
        // precompute: simdgroup backend (M1-M4) and MXU backend (M5 default).
        // (out atol 2e-3, state atol 1e-3, rtol 5e-2).
        let configs: [(&str, &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bf16>, Vec<f32>, Vec<f32>), u32, bool); 2] =
            [("simd_vt32", &precompute_simd, 32, false), ("mxu_vt32", &precompute_simd, 32, true)];

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
    // Includes the production router's threshold lengths (256 -> MXU path,
    // 1024 -> simd path) so the exact chunked chain encoded by
    // `DeltaNetMixer::run_delta_rule_prefill_chunked` is validated against the
    // recurrent path at those trigger points for BOTH megaApply backends.
    for suffix_len in [1usize, 63, 64, 65, 200, 256, 512, 1024, 4096] {
        mode_l_pipeline_vs_recurrent_impl(suffix_len);
    }
}

// Live-device routing: drives the production router `select_gdn_prefill_path`
// with the real Metal `Context` predicate (`supports_mxu`) and checks it picks
// the path the dispatch table prescribes for this machine's tier. The simd
// route is intentionally left to the separate dynamic-caching capability PR.
#[uzu_test]
fn test_gdn_prefill_router_live_context() {
    use crate::encodable_block::{CHUNKED_MXU_MIN_T, GdnPrefillPath, select_gdn_prefill_path};

    let context = <Metal as Backend>::Context::new().expect("context");
    let mxu = context.supports_mxu();
    eprintln!("ROUTER_LIVE supports_mxu={mxu}");

    let route = |t: usize| select_gdn_prefill_path(context.as_ref(), t);

    // Below the smallest chunked threshold, always recurrent regardless of tier.
    assert_eq!(route(1), GdnPrefillPath::Recurrent);
    assert_eq!(route(CHUNKED_MXU_MIN_T - 1), GdnPrefillPath::Recurrent);

    let expected_at = |t: usize| {
        if mxu && t >= CHUNKED_MXU_MIN_T {
            GdnPrefillPath::ChunkedModeL
        } else {
            GdnPrefillPath::Recurrent
        }
    };
    for &t in &[CHUNKED_MXU_MIN_T, 512usize, 1024, 4096] {
        assert_eq!(route(t), expected_at(t), "T={t}");
    }
}

// End-to-end prefill sweep at the SAME scope (precompute + apply + NormGate):
// recurrent, Mode-L VT32 simdgroup, and Mode-L VT32 MXU when the device supports
// it. Each path warms >=500 ms before timing. The MXU PSO is constructed only
// behind `context.supports_mxu()`, so this still runs on MXU-less machines.
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

    eprintln!("\n=== DeltaNet FLEET Mode L vs Recurrent (end-to-end prefill) ===");
    eprintln!("  C={chunk_size} HV={num_v_heads} HK={num_k_heads} K={head_k_dim} V={head_v_dim} dtype=bf16");
    eprintln!("  scope: precompute + apply + NormGate for every path");
    eprintln!("FLEETBENCH\tT\tpath\tmedian_ms\tmin_ms\tmax_ms\tstd_ms");

    // (T, recurrent_ms, vt32_simd_ms, vt32_mxu_ms) collected for the final table.
    let mut summary: Vec<(usize, f64, f64, Option<f64>)> = Vec::new();

    for suffix_len in [256usize, 512, 1024, 4096, 32768] {
        let num_chunks = suffix_len.div_ceil(chunk_size);

        let in_proj: Vec<bf16> =
            (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
        let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
        let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

        let context = <Metal as Backend>::Context::new().expect("context");
        let in_proj_array = alloc_allocation_with_data::<Metal, _>(&context, &in_proj);
        let a_log_array = alloc_allocation_with_data::<Metal, _>(&context, &a_log);
        let dt_bias_array = alloc_allocation_with_data::<Metal, _>(&context, &dt_bias);
        let norm_weight_array = alloc_allocation_with_data::<Metal, _>(&context, &norm_weight);
        let alloc = |data_type, len| alloc_dtype::<Metal>(&context, len, data_type);

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

        let supports_mxu = context.supports_mxu();

        // Kernels. Gram/SolveT use the single f32 simdgroup precompute path;
        // MegaApply is timed with USE_MXU=false everywhere, and with USE_MXU=true
        // only on devices that support it.
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
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64).unwrap();
        // solveT USE_MXU=false (simdgroup).
        let solve_t_simd =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32).unwrap();
        // megaApply VT=32, USE_MXU=false (simdgroup) — the M1-M4 candidate.
        let mega_vt32_simd = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            32,
            false,
        )
        .unwrap();
        let mega_vt32_mxu = if supports_mxu {
            Some(
                <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
                    &context,
                    DataType::BF16,
                    DataType::BF16,
                    32,
                    true,
                )
                .unwrap(),
            )
        } else {
            None
        };

        macro_rules! encode_mode_l {
            ($mega:expr) => {{
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encoder.encode_fill(&mut mode_l_state, 0);
                prep_k.encode(
                    &in_proj_array,
                    &a_log_array,
                    &dt_bias_array,
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
                    &in_proj_array,
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
                &in_proj_array,
                &a_log_array,
                &dt_bias_array,
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
                &in_proj_array,
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

        let vt32_mxu_res = if let Some(mega_vt32_mxu) = mega_vt32_mxu.as_ref() {
            let mut run_vt32_mxu = || encode_mode_l!(mega_vt32_mxu);
            warmup(&mut run_vt32_mxu);
            let result = run_perf_with_warmup("mode_l_vt32_mxu", 3, iterations, &mut run_vt32_mxu);
            print_row("mode_l_vt32_mxu", &result);
            Some(result)
        } else {
            eprintln!("FLEETBENCH\t{}\tmode_l_vt32_mxu\tn/a\tn/a\tn/a\tn/a", suffix_len);
            None
        };

        eprintln!(
            "FLEETVERDICT\tT={}\trec={:.3}\tvt32_simd={:.3}\tvt32s_vs_rec={:.3}x\tvt32_mxu={}",
            suffix_len,
            rec_res.median_ms,
            vt32_simd_res.median_ms,
            rec_res.median_ms / vt32_simd_res.median_ms,
            vt32_mxu_res
                .as_ref()
                .map(|r| format!("{:.3} ({:.3}x)", r.median_ms, rec_res.median_ms / r.median_ms))
                .unwrap_or_else(|| "n/a".to_string()),
        );

        summary.push((suffix_len, rec_res.median_ms, vt32_simd_res.median_ms, vt32_mxu_res.map(|r| r.median_ms)));
    }

    // Final clean table across all T.
    eprintln!("\n=== FLEET Mode L vs Recurrent summary (median ms) ===");
    eprintln!(
        "{:>7}  {:>11}  {:>11}  {:>12}  {:>11}  {:>12}",
        "T", "recurrent", "vt32-simd", "simd/rec", "vt32-mxu", "mxu/rec"
    );
    for (t, rec, vt32, mxu) in &summary {
        let mxu_ms = mxu.map(|value| format!("{value:.3}")).unwrap_or_else(|| "n/a".to_string());
        let mxu_speedup = mxu.map(|value| format!("{:.3}x", rec / value)).unwrap_or_else(|| "n/a".to_string());
        eprintln!(
            "{:>7}  {:>11.3}  {:>11.3}  {:>11.3}x  {:>11}  {:>12}",
            t,
            rec,
            vt32,
            rec / vt32,
            mxu_ms,
            mxu_speedup
        );
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
        let in_proj_array = alloc_allocation_with_data::<Metal, _>(&context, &in_proj);
        let a_log_array = alloc_allocation_with_data::<Metal, _>(&context, &a_log);
        let dt_bias_array = alloc_allocation_with_data::<Metal, _>(&context, &dt_bias);
        let norm_weight_array = alloc_allocation_with_data::<Metal, _>(&context, &norm_weight);

        let alloc = |data_type, len| alloc_dtype::<Metal>(&context, len, data_type);

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
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(&context, 128, 64).unwrap();
        let solve_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(&context, 64, false).unwrap();
        let solve_t_k =
            <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(&context, 64, 32).unwrap();
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
                &in_proj_array,
                &a_log_array,
                &dt_bias_array,
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
                &in_proj_array,
                &a_log_array,
                &dt_bias_array,
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
                    &in_proj_array,
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
            };
            warmup(&mut run);
            run_perf_with_warmup("norm_gate", 3, iterations, &mut run)
        };

        macro_rules! encode_precompute {
            ($encoder:expr) => {{
                prep_k.encode(
                    &in_proj_array,
                    &a_log_array,
                    &dt_bias_array,
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
                    &in_proj_array,
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
