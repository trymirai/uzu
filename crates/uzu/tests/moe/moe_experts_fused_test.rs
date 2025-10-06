#![cfg(feature = "moe_dev_tests")]

use half::bf16;
use metal::MTLResourceOptions;
use uzu::backends::metal::{
    MTLContext,
    kernel::{KernelDataType, MoeExpertsArguments, MoeExpertsKernel},
};

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

fn silu(x: f32, alpha: f32) -> f32 {
    x / (1.0 + (-alpha * x).exp())
}

fn gelu(x: f32) -> f32 {
    const K0: f32 = 0.7978845608;
    const K1: f32 = 0.044715;
    if x > 10.0 {
        return x;
    }
    if x < -10.0 {
        return 0.0;
    }
    let x3 = x * x * x;
    0.5 * x * (1.0 + (K0 * (x + K1 * x3)).tanh())
}

fn cpu_expert_forward(
    x: &[f32],
    w13: &[f32],
    w2: &[f32],
    up_bias: &[f32],
    down_bias: &[f32],
    gating_code: u32,
    silu_alpha: f32,
    gate_clip: (f32, f32),
    up_clip: (f32, f32),
    d_model: usize,
    d_ff: usize,
) -> Vec<f32> {
    let mut up = vec![0.0f32; d_ff];
    let mut gate = vec![0.0f32; d_ff];
    for ff in 0..d_ff {
        let mut up_sum = up_bias[ff];
        let mut gate_sum = up_bias[d_ff + ff];
        for dm in 0..d_model {
            let base = dm * 2 * d_ff;
            let x_val = x[dm];
            up_sum += x_val * w13[base + ff];
            gate_sum += x_val * w13[base + d_ff + ff];
        }
        up[ff] = up_sum.clamp(up_clip.0, up_clip.1);
        gate[ff] = gate_sum.clamp(gate_clip.0, gate_clip.1);
    }

    let mut hidden = vec![0.0f32; d_ff];
    for i in 0..d_ff {
        hidden[i] = match gating_code {
            0 => gelu(up[i]),
            1 => silu(up[i], silu_alpha),
            2 => silu(gate[i], silu_alpha) * up[i],
            3 => gelu(gate[i]) * up[i],
            _ => silu(gate[i], silu_alpha) * up[i],
        };
    }

    let mut out = vec![0.0f32; d_model];
    for dm in 0..d_model {
        let mut sum = down_bias[dm];
        for ff in 0..d_ff {
            sum += hidden[ff] * w2[ff * d_model + dm];
        }
        out[dm] = sum;
    }
    out
}

fn run_decode_parity_case(
    ctx: &MTLContext,
    dtype: KernelDataType,
    gating_code: u32,
) -> f32 {
    let t = 1usize;
    let k = 2usize;
    let e = 4usize;
    let d_model = 8usize;
    let d_ff = 6usize;
    let silu_alpha = 1.702f32;
    let gate_clip = (-6.0f32, 7.0f32);
    let up_clip = (-6.0f32, 8.0f32);

    let x: Vec<f32> = vec![
        0.25, -0.5, 1.0, -1.25, 0.75, -0.375, 0.125, 0.875,
    ];
    let active_experts = [1usize, 3usize];
    let sum_k = active_experts.len();

    let mut x_perm = Vec::with_capacity(sum_k * d_model);
    for _ in &active_experts {
        x_perm.extend_from_slice(&x);
    }

    let mut expert_offsets = vec![0u32; e + 1];
    let mut running = 0u32;
    for expert in 0..e {
        expert_offsets[expert] = running;
        let count = active_experts
            .iter()
            .filter(|&&idx| idx == expert)
            .count() as u32;
        running += count;
    }
    expert_offsets[e] = running;

    let elem_w13 = e * d_model * 2 * d_ff;
    let elem_w2 = e * d_ff * d_model;
    let elem_up_bias = e * 2 * d_ff;
    let elem_down_bias = e * d_model;

    let mut base_w13 = vec![0.0f32; elem_w13];
    for (expert, chunk) in base_w13.chunks_mut(d_model * 2 * d_ff).enumerate() {
        for dm in 0..d_model {
            for ch in 0..(2 * d_ff) {
                let idx = dm * 2 * d_ff + ch;
                chunk[idx] = (expert as f32 + 1.0) * 0.05
                    + dm as f32 * 0.003
                    + ch as f32 * 0.001;
            }
        }
    }

    let mut base_w2 = vec![0.0f32; elem_w2];
    for (expert, chunk) in base_w2.chunks_mut(d_ff * d_model).enumerate() {
        for ff in 0..d_ff {
            for dm in 0..d_model {
                let idx = ff * d_model + dm;
                chunk[idx] = (expert as f32 + 1.0) * 0.04
                    + ff as f32 * 0.002
                    + dm as f32 * 0.001;
            }
        }
    }

    let mut base_up_bias = vec![0.0f32; elem_up_bias];
    for (expert, chunk) in base_up_bias.chunks_mut(2 * d_ff).enumerate() {
        for ch in 0..(2 * d_ff) {
            chunk[ch] = (expert as f32 + 1.0) * 0.03 + ch as f32 * 0.001;
        }
    }

    let mut base_down_bias = vec![0.0f32; elem_down_bias];
    for (expert, chunk) in base_down_bias.chunks_mut(d_model).enumerate() {
        for dm in 0..d_model {
            chunk[dm] = (expert as f32 + 1.0) * 0.02 + dm as f32 * 0.001;
        }
    }

    let (x_perm_upload_bytes, x_perm_buf, x_perm_cpu): (u64, metal::Buffer, Vec<f32>) =
        match dtype {
            KernelDataType::Float32 => {
                let bytes = (x_perm.len() * std::mem::size_of::<f32>()) as u64;
                let buf = ctx.device.new_buffer_with_data(
                    x_perm.as_ptr() as *const _,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                (bytes, buf, x_perm.clone())
            },
            KernelDataType::BFloat16 => {
                let x_bf16: Vec<bf16> = x_perm.iter().map(|&v| bf16::from_f32(v)).collect();
                let bytes = (x_bf16.len() * std::mem::size_of::<bf16>()) as u64;
                let buf = ctx.device.new_buffer_with_data(
                    x_bf16.as_ptr() as *const _,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                let cpu = x_bf16.iter().map(|&v| f32::from(v)).collect();
                (bytes, buf, cpu)
            },
            _ => panic!("dtype {:?} not supported", dtype),
        };
    let _ = x_perm_upload_bytes;

    let upload_as = |data: &[f32], dtype: KernelDataType| -> (metal::Buffer, Vec<f32>) {
        match dtype {
            KernelDataType::Float32 => {
                let bytes = (data.len() * std::mem::size_of::<f32>()) as u64;
                let buf = ctx.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                (buf, data.to_vec())
            },
            KernelDataType::BFloat16 => {
                let bf16_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
                let bytes = (bf16_data.len() * std::mem::size_of::<bf16>()) as u64;
                let buf = ctx.device.new_buffer_with_data(
                    bf16_data.as_ptr() as *const _,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                let cpu = bf16_data.iter().map(|&v| f32::from(v)).collect();
                (buf, cpu)
            },
            _ => panic!("dtype {:?} not supported", dtype),
        }
    };

    let (w13_buf, w13_cpu) = upload_as(&base_w13, dtype);
    let (w2_buf, w2_cpu) = upload_as(&base_w2, dtype);
    let (up_biases_buf, up_biases_cpu) = upload_as(&base_up_bias, dtype);
    let (down_biases_buf, down_biases_cpu) = upload_as(&base_down_bias, dtype);

    let y_partial_buf = match dtype {
        KernelDataType::Float32 => ctx.device.new_buffer(
            (sum_k * d_model * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        KernelDataType::BFloat16 => ctx.device.new_buffer(
            (sum_k * d_model * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        _ => unreachable!(),
    };

    let offsets_buf = ctx.device.new_buffer_with_data(
        expert_offsets.as_ptr() as *const _,
        (expert_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let zero_u32 = vec![0u32; e.max(1)];
    let tile_counts_buf = ctx.device.new_buffer_with_data(
        zero_u32.as_ptr() as *const _,
        (zero_u32.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tile_row_offsets_buf = ctx.device.new_buffer_with_data(
        expert_offsets.as_ptr() as *const _,
        (expert_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tile_map_buf = ctx.device.new_buffer(
        (3 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let total_tiles_buf = ctx.device.new_buffer_with_data(
        [0u32; 2].as_ptr() as *const _,
        (2 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dispatch_args_buf = ctx.device.new_buffer_with_data(
        [0u32; 3].as_ptr() as *const _,
        (3 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let experts = MoeExpertsKernel::new(ctx).expect("experts kernel");
    let num_tiles_n = (d_model + 63) / 64;

    let cb = ctx.command_queue.new_command_buffer();
    experts
        .encode(
            &cb,
            MoeExpertsArguments {
                x_perm_buffer: &x_perm_buf,
                expert_offsets: &offsets_buf,
                w13_all: &w13_buf,
                w2_all: &w2_buf,
                y_partial: &y_partial_buf,
                up_biases: &up_biases_buf,
                down_biases: &down_biases_buf,
                tile_counts: &tile_counts_buf,
                tile_row_offsets: &tile_row_offsets_buf,
                tile_map: &tile_map_buf,
                total_tiles: &total_tiles_buf,
                dispatch_args: &dispatch_args_buf,
                num_tiles_n,
                t,
                d_model,
                d_ff,
                e,
                k,
                gating_code,
                gate_clip_min: gate_clip.0,
                gate_clip_max: gate_clip.1,
                up_clip_min: up_clip.0,
                up_clip_max: up_clip.1,
                silu_alpha,
                data_type: dtype,
            },
        )
        .expect("experts encode");
    cb.commit();
    cb.wait_until_completed();

    let y_gpu: Vec<f32> = match dtype {
        KernelDataType::Float32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    y_partial_buf.contents() as *const f32,
                    sum_k * d_model,
                )
            };
            slice.to_vec()
        },
        KernelDataType::BFloat16 => {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    y_partial_buf.contents() as *const bf16,
                    sum_k * d_model,
                )
            };
            slice.iter().map(|&v| f32::from(v)).collect()
        },
        _ => unreachable!(),
    };

    let mut expected = Vec::with_capacity(sum_k * d_model);
    for (row_idx, &expert_idx) in active_experts.iter().enumerate() {
        let base = expert_idx;
        let w13_slice = &w13_cpu[base * d_model * 2 * d_ff..(base + 1) * d_model * 2 * d_ff];
        let w2_slice = &w2_cpu[base * d_ff * d_model..(base + 1) * d_ff * d_model];
        let up_bias_slice = &up_biases_cpu[base * 2 * d_ff..(base + 1) * 2 * d_ff];
        let down_bias_slice = &down_biases_cpu[base * d_model..(base + 1) * d_model];

        let y = cpu_expert_forward(
            &x_perm_cpu[row_idx * d_model..(row_idx + 1) * d_model],
            w13_slice,
            w2_slice,
            up_bias_slice,
            down_bias_slice,
            gating_code,
            silu_alpha,
            gate_clip,
            up_clip,
            d_model,
            d_ff,
        );
        expected.extend_from_slice(&y);
    }

    let mut max_abs = 0.0f32;
    let tol = match dtype {
        KernelDataType::Float32 => 1e-4f32,
        KernelDataType::BFloat16 => 3e-3f32,
        _ => unreachable!(),
    };

    for i in 0..expected.len() {
        max_abs = max_abs.max((y_gpu[i] - expected[i]).abs());
    }

    assert!(
        max_abs < tol,
        "max diff {} exceeds tol {} (dtype {:?}, gating {})",
        max_abs,
        tol,
        dtype,
        gating_code
    );

    max_abs
}

#[test]
fn test_moe_experts_decode_parity() {
    let ctx = create_ctx();
    for &dtype in &[KernelDataType::Float32, KernelDataType::BFloat16] {
        for gating_code in 0u32..=3u32 {
            run_decode_parity_case(&ctx, dtype, gating_code);
        }
    }
}
