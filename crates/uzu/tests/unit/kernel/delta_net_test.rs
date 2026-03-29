#![cfg(metal_backend)]

use std::ops::{Deref, DerefMut};

use uzu::{
    ArrayContextExt, DataType,
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

    let w_array = context.create_array_from(&[w.len()], w, "w");
    let b_array = context.create_array_from(&[b.len()], b, "b");
    let in_out_array = context.create_array_from(&[in_proj.len()], in_proj, "in_out");
    let state_array = context.create_array_from(&[state.len()], state, "state");

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(&context, DataType::F32, true)
        .expect("Failed to create kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        w_array.buffer().borrow().deref(),
        Some(b_array.buffer().borrow().deref()),
        in_out_array.buffer().borrow_mut().deref_mut(),
        state_array.buffer().borrow_mut().deref_mut(),
        kernel_size,
        conv_dim,
        state_stride,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = in_out_array.as_slice::<f32>()[..conv_dim as usize].to_vec();
    let new_state = state_array.as_slice::<f32>().to_vec();
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

    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj, "in_proj");
    let a_log_array = context.create_array_from(&[a_log.len()], a_log, "a_log");
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias, "dt_bias");
    let norm_weight_array = context.create_array_from(&[norm_weight.len()], norm_weight, "norm_weight");
    let state_array = context.create_array_from(&[state.len()], state, "state");
    let out_array = context.create_array_zeros(&[value_dim as usize], DataType::F32, "out");

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetUpdateKernel::new(&context, DataType::F32, head_k_dim)
        .expect("Failed to create kernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        in_proj_array.buffer().borrow().deref(),
        a_log_array.buffer().borrow().deref(),
        dt_bias_array.buffer().borrow().deref(),
        norm_weight_array.buffer().borrow().deref(),
        state_array.buffer().borrow_mut().deref_mut(),
        out_array.buffer().borrow_mut().deref_mut(),
        num_v_heads,
        num_k_heads,
        head_v_dim,
        key_dim,
        value_dim,
        1e-6f32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let out = out_array.as_slice::<f32>().to_vec();
    let new_state = state_array.as_slice::<f32>().to_vec();
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

#[test]
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

#[test]
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
    let in_proj: Vec<f32> = (0..suffix_len * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();

    // Reference: decode conv token-by-token
    let mut ref_state = init_state.clone();
    let mut ref_outputs = vec![0.0f32; suffix_len * conv_dim];
    for t in 0..suffix_len {
        let token_in: Vec<f32> = in_proj[t * total_proj_dim..t * total_proj_dim + conv_dim].to_vec();
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

    // Test: Conv1dPack + DeltaNetConvScan on Metal
    let context = <Metal as Backend>::Context::new().expect("context");
    let state_array = context.create_array_from(&[init_state.len()], &init_state, "state");
    let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj, "in_proj");
    let w_array = context.create_array_from(&[w.len()], &w, "w");
    let b_array = context.create_array_from(&[b.len()], &b, "b");

    let padded_len = (tap_count + suffix_len) * total_proj_dim;
    let padded_array = context.create_array_zeros(&[padded_len], DataType::F32, "padded");
    let state_out_array = context.create_array_zeros(&[conv_dim * tap_count], DataType::F32, "state_out");

    let pack_kernel =
        <<Metal as Backend>::Kernels as Kernels>::Conv1dPackKernel::new(&context, DataType::F32).expect("pack");
    let scan_kernel =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetConvScanKernel::new(&context, DataType::F32, true)
            .expect("scan");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    pack_kernel.encode(
        state_array.buffer().borrow().deref(),
        in_proj_array.buffer().borrow().deref(),
        padded_array.buffer().borrow_mut().deref_mut(),
        tap_count as u32,
        total_proj_dim as u32,
        suffix_len as u32,
        conv_dim as u32,
        &mut encoder,
    );
    scan_kernel.encode(
        padded_array.buffer().borrow().deref(),
        w_array.buffer().borrow().deref(),
        Some(b_array.buffer().borrow().deref()),
        in_proj_array.buffer().borrow_mut().deref_mut(),
        state_out_array.buffer().borrow_mut().deref_mut(),
        suffix_len as u32,
        kernel_size as u32,
        total_proj_dim as u32,
        tap_count as u32,
        conv_dim as u32,
        total_proj_dim as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let in_proj_result = in_proj_array.as_slice::<f32>();
    let mut scan_outputs = vec![0.0f32; suffix_len * conv_dim];
    for t in 0..suffix_len {
        scan_outputs[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&in_proj_result[t * total_proj_dim..t * total_proj_dim + conv_dim]);
    }
    let scan_state = state_out_array.as_slice::<f32>().to_vec();

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

#[test]
fn test_delta_net_update_qwen35_shapes() {
    test_delta_net_update_impl(48, 16, 128, 128, "DeltaNetUpdate Qwen3.5");
}

// DeltaNetPrefill + NormGate

fn run_prefill_with_norm_gate(
    in_proj: &[f32],
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
    let num_dv_groups = (head_v_dim as u32 + 7) / 8;

    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], in_proj, "in_proj");
    let a_log_array = context.create_array_from(&[a_log.len()], a_log, "a_log");
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], dt_bias, "dt_bias");
    let norm_weight_array = context.create_array_from(&[norm_weight.len()], norm_weight, "norm_weight");
    let state_array = context.create_array_from(&[state.len()], state, "state");
    let out_array = context.create_array_zeros(&[suffix_len * value_dim], DataType::F32, "out");
    let q_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "q_norm");
    let k_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "k_norm");

    let beta_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "beta");
    let decay_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "decay");

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

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    prep_k.encode(
        in_proj_array.buffer().borrow().deref(),
        a_log_array.buffer().borrow().deref(),
        dt_bias_array.buffer().borrow().deref(),
        q_norm_array.buffer().borrow_mut().deref_mut(),
        k_norm_array.buffer().borrow_mut().deref_mut(),
        beta_array.buffer().borrow_mut().deref_mut(),
        decay_array.buffer().borrow_mut().deref_mut(),
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    prefill_k.encode(
        q_norm_array.buffer().borrow().deref(),
        k_norm_array.buffer().borrow().deref(),
        beta_array.buffer().borrow().deref(),
        decay_array.buffer().borrow().deref(),
        in_proj_array.buffer().borrow().deref(),
        state_array.buffer().borrow_mut().deref_mut(),
        out_array.buffer().borrow_mut().deref_mut(),
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
        out_array.buffer().borrow_mut().deref_mut(),
        in_proj_array.buffer().borrow().deref(),
        norm_weight_array.buffer().borrow().deref(),
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

    (out_array.as_slice::<f32>().to_vec(), state_array.as_slice::<f32>().to_vec())
}

fn test_prefill_norm_gate_impl(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    suffix_len: usize,
    label: &str,
) {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let state_size = num_v_heads * head_k_dim * head_v_dim;

    let in_proj: Vec<f32> = (0..suffix_len * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();
    let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.05).collect();
    let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.3 + (i as f32) * 0.02).collect();
    let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();
    let state: Vec<f32> = (0..state_size).map(|i| ((i % 29) as f32) * 0.005 - 0.05).collect();

    // Reference: fused decode kernel token-by-token
    let mut ref_state = state.clone();
    let mut ref_outputs = vec![0.0f32; suffix_len * value_dim];
    for t in 0..suffix_len {
        let token_in: Vec<f32> = in_proj[t * total_proj_dim..(t + 1) * total_proj_dim].to_vec();
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
        ref_outputs[t * value_dim..(t + 1) * value_dim].copy_from_slice(&out);
    }

    let (gpu_out, gpu_state) = run_prefill_with_norm_gate(
        &in_proj,
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

    assert_close(&ref_outputs, &gpu_out, 1e-3, 1e-2, &format!("{label} output"));
    assert_close(&ref_state, &gpu_state, 1e-3, 1e-2, &format!("{label} state"));
}

#[test]
fn test_delta_net_prefill_qwen35_shapes() {
    test_prefill_norm_gate_impl(48, 16, 128, 128, 32, "Prefill+NormGate Qwen3.5");
}

#[test]
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
    let cpu_in_proj = cpu_ctx.create_array_from(&[in_proj.len()], &in_proj, "in_proj");
    let cpu_a_log = cpu_ctx.create_array_from(&[a_log.len()], &a_log, "a_log");
    let cpu_dt_bias = cpu_ctx.create_array_from(&[dt_bias.len()], &dt_bias, "dt_bias");
    let cpu_q = cpu_ctx.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "q");
    let cpu_k = cpu_ctx.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "k");
    let cpu_beta = cpu_ctx.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "beta");
    let cpu_decay = cpu_ctx.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "decay");

    let cpu_prep = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &cpu_ctx,
        DataType::F32,
        head_k_dim as u32,
    )
    .unwrap();
    let mut cpu_enc = Encoder::new(cpu_ctx.as_ref()).expect("encoder");
    cpu_prep.encode(
        cpu_in_proj.buffer().borrow().deref(),
        cpu_a_log.buffer().borrow().deref(),
        cpu_dt_bias.buffer().borrow().deref(),
        cpu_q.buffer().borrow_mut().deref_mut(),
        cpu_k.buffer().borrow_mut().deref_mut(),
        cpu_beta.buffer().borrow_mut().deref_mut(),
        cpu_decay.buffer().borrow_mut().deref_mut(),
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut cpu_enc,
    );
    cpu_enc.end_encoding().submit().wait_until_completed().unwrap();

    let ref_q = cpu_q.as_slice::<f32>().to_vec();
    let ref_k = cpu_k.as_slice::<f32>().to_vec();
    let ref_beta = cpu_beta.as_slice::<f32>().to_vec();
    let ref_decay = cpu_decay.as_slice::<f32>().to_vec();

    // Metal
    let context = <Metal as Backend>::Context::new().expect("context");
    let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj, "in_proj");
    let a_log_array = context.create_array_from(&[a_log.len()], &a_log, "a_log");
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias, "dt_bias");
    let q_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "q_norm");
    let k_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "k_norm");

    let beta_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "beta");
    let decay_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "decay");

    let prep_k = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::F32,
        head_k_dim as u32,
    )
    .unwrap();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    prep_k.encode(
        in_proj_array.buffer().borrow().deref(),
        a_log_array.buffer().borrow().deref(),
        dt_bias_array.buffer().borrow().deref(),
        q_norm_array.buffer().borrow_mut().deref_mut(),
        k_norm_array.buffer().borrow_mut().deref_mut(),
        beta_array.buffer().borrow_mut().deref_mut(),
        decay_array.buffer().borrow_mut().deref_mut(),
        num_v_heads as u32,
        num_k_heads as u32,
        key_dim as u32,
        value_dim as u32,
        suffix_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let gpu_q = q_norm_array.as_slice::<f32>();
    let gpu_k = k_norm_array.as_slice::<f32>();
    let gpu_beta = beta_array.as_slice::<f32>();
    let gpu_decay = decay_array.as_slice::<f32>();

    assert_close(gpu_q, &ref_q, 1e-4, 1e-3, "prep q_norm");
    assert_close(gpu_k, &ref_k, 1e-4, 1e-3, "prep k_norm");
    assert_close(gpu_beta, &ref_beta, 1e-4, 1e-3, "prep beta");
    assert_close(gpu_decay, &ref_decay, 1e-4, 1e-3, "prep decay");
}

#[test]
#[ignore]
fn bench_delta_net_prefill() {
    use crate::common::perf::run_perf_with_warmup;

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
    let in_proj_array = context.create_array_from(&[in_proj.len()], &in_proj, "in_proj");
    let a_log_array = context.create_array_from(&[a_log.len()], &a_log, "a_log");
    let dt_bias_array = context.create_array_from(&[dt_bias.len()], &dt_bias, "dt_bias");
    let norm_weight_array = context.create_array_from(&[norm_weight.len()], &norm_weight, "norm_weight");
    let out_array = context.create_array_zeros(&[suffix_len * value_dim], DataType::F32, "out");
    let q_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "q_norm");
    let k_norm_array = context.create_array_zeros(&[suffix_len * key_dim], DataType::F32, "k_norm");

    let beta_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "beta");
    let decay_array = context.create_array_zeros(&[suffix_len * num_v_heads], DataType::F32, "decay");

    let num_dv_groups = (head_v_dim as u32 + 7) / 8;

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
            in_proj_array.buffer().borrow().deref(),
            a_log_array.buffer().borrow().deref(),
            dt_bias_array.buffer().borrow().deref(),
            q_norm_array.buffer().borrow_mut().deref_mut(),
            k_norm_array.buffer().borrow_mut().deref_mut(),
            beta_array.buffer().borrow_mut().deref_mut(),
            decay_array.buffer().borrow_mut().deref_mut(),
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
    let mut state_array = context.create_array_zeros(&[state_size], DataType::F32, "state");

    let prefill_result = run_perf_with_warmup("prep+prefill+norm_gate", 5, 50, || {
        state_array.as_slice_mut::<f32>().fill(0.0);

        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        prep_k.encode(
            in_proj_array.buffer().borrow().deref(),
            a_log_array.buffer().borrow().deref(),
            dt_bias_array.buffer().borrow().deref(),
            q_norm_array.buffer().borrow_mut().deref_mut(),
            k_norm_array.buffer().borrow_mut().deref_mut(),
            beta_array.buffer().borrow_mut().deref_mut(),
            decay_array.buffer().borrow_mut().deref_mut(),
            num_v_heads as u32,
            num_k_heads as u32,
            key_dim as u32,
            value_dim as u32,
            suffix_len as u32,
            &mut encoder,
        );
        prefill_k.encode(
            q_norm_array.buffer().borrow().deref(),
            k_norm_array.buffer().borrow().deref(),
            beta_array.buffer().borrow().deref(),
            decay_array.buffer().borrow().deref(),
            in_proj_array.buffer().borrow().deref(),
            state_array.buffer().borrow_mut().deref_mut(),
            out_array.buffer().borrow_mut().deref_mut(),
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
            out_array.buffer().borrow_mut().deref_mut(),
            in_proj_array.buffer().borrow().deref(),
            norm_weight_array.buffer().borrow().deref(),
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
