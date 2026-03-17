#![cfg(target_os = "macos")]

use std::ops::{Deref, DerefMut};

use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels,
            kernel::{DeltaNetConvUpdateKernel, DeltaNetUpdateKernel},
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

    let mut cb = context.create_command_buffer().expect("cb").start_encoding();
    kernel.encode(
        w_array.buffer().borrow().deref(),
        Some(b_array.buffer().borrow().deref()),
        in_out_array.buffer().borrow_mut().deref_mut(),
        state_array.buffer().borrow_mut().deref_mut(),
        kernel_size,
        conv_dim,
        state_stride,
        &mut cb,
    );
    cb.end_encoding().submit().wait_until_completed().unwrap();

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
    let out_array = context.create_array(&[value_dim as usize], DataType::F32, "out");

    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetUpdateKernel::new(&context, DataType::F32)
        .expect("Failed to create kernel");

    let mut cb = context.create_command_buffer().expect("cb").start_encoding();
    kernel.encode(
        in_proj_array.buffer().borrow().deref(),
        a_log_array.buffer().borrow().deref(),
        dt_bias_array.buffer().borrow().deref(),
        norm_weight_array.buffer().borrow().deref(),
        state_array.buffer().borrow_mut().deref_mut(),
        out_array.buffer().borrow_mut().deref_mut(),
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        1e-6f32,
        &mut cb,
    );
    cb.end_encoding().submit().wait_until_completed().unwrap();

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
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut worst_idx = 0;
    for i in 0..a.len() {
        let abs_diff = (a[i] - b[i]).abs();
        let denom = a[i].abs().max(b[i].abs()).max(1e-8);
        let rel_diff = abs_diff / denom;
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            worst_idx = i;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
    }
    println!(
        "{label}: max_abs_diff={max_abs_diff:.6e} max_rel_diff={max_rel_diff:.6e} worst_idx={worst_idx} (cpu={:.6} gpu={:.6})",
        a[worst_idx], b[worst_idx]
    );
    assert!(
        max_abs_diff < atol || max_rel_diff < rtol,
        "{label}: FAILED — max_abs_diff={max_abs_diff:.6e} (atol={atol:.6e}), max_rel_diff={max_rel_diff:.6e} (rtol={rtol:.6e})"
    );
}

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
fn test_delta_net_update_small() {
    test_delta_net_update_impl(4, 2, 16, 16, "DeltaNetUpdate small");
}

#[test]
fn test_delta_net_update_qwen35_shapes() {
    test_delta_net_update_impl(48, 16, 128, 128, "DeltaNetUpdate Qwen3.5");
}
