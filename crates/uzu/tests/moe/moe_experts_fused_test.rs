use half::bf16;
use metal::MTLResourceOptions;
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::moe::{
        MoeExpertsArguments, MoeExpertsKernel, MoeExpertsTwoPassArguments,
    },
};

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

fn silu(
    x: f32,
    alpha: f32,
) -> f32 {
    x / (1.0 + (-alpha * x).exp())
}

fn bf16_add(
    lhs: bf16,
    rhs: bf16,
) -> bf16 {
    bf16::from_f32(lhs.to_f32() + rhs.to_f32())
}

fn bf16_mul(
    lhs: bf16,
    rhs: bf16,
) -> bf16 {
    bf16::from_f32(lhs.to_f32() * rhs.to_f32())
}

fn bf16_clamp(
    val: bf16,
    min: f32,
    max: f32,
) -> bf16 {
    bf16::from_f32(val.to_f32().clamp(min, max))
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
    x: &[bf16],
    w13: &[bf16],
    w2: &[bf16],
    up_bias: &[bf16],
    down_bias: &[bf16],
    gating_code: u32,
    silu_alpha: f32,
    gate_clip: (f32, f32),
    up_clip: (f32, f32),
    d_model: usize,
    d_ff: usize,
) -> Vec<f32> {
    let mut up = vec![bf16::from_f32(0.0); d_ff];
    let mut gate = vec![bf16::from_f32(0.0); d_ff];
    for ff in 0..d_ff {
        let mut up_sum = up_bias[ff];
        let mut gate_sum = up_bias[d_ff + ff];
        for dm in 0..d_model {
            let base = dm * 2 * d_ff;
            up_sum = bf16_add(up_sum, bf16_mul(x[dm], w13[base + ff]));
            gate_sum =
                bf16_add(gate_sum, bf16_mul(x[dm], w13[base + d_ff + ff]));
        }
        up[ff] = bf16_clamp(up_sum, up_clip.0, up_clip.1);
        gate[ff] = bf16_clamp(gate_sum, gate_clip.0, gate_clip.1);
    }

    let mut hidden = vec![bf16::from_f32(0.0); d_ff];
    for i in 0..d_ff {
        hidden[i] = match gating_code {
            0 => bf16::from_f32(gelu(up[i].to_f32())),
            1 => bf16::from_f32(silu(up[i].to_f32(), silu_alpha)),
            2 => bf16_mul(
                bf16::from_f32(silu(gate[i].to_f32(), silu_alpha)),
                up[i],
            ),
            3 => bf16_mul(bf16::from_f32(gelu(gate[i].to_f32())), up[i]),
            _ => bf16_mul(
                bf16::from_f32(silu(gate[i].to_f32(), silu_alpha)),
                up[i],
            ),
        };
    }

    let mut out = vec![0.0f32; d_model];
    for dm in 0..d_model {
        let mut sum = down_bias[dm];
        for ff in 0..d_ff {
            sum = bf16_add(sum, bf16_mul(hidden[ff], w2[ff * d_model + dm]));
        }
        out[dm] = sum.to_f32();
    }
    out
}

fn run_decode_parity_case(
    ctx: &MTLContext,
    dtype: KernelDataType,
    gating_code: u32,
    force_two_pass: bool,
) -> f32 {
    use std::mem::size_of;

    const K_TILE: usize = 64;

    let t = 1usize;
    let e = 4usize;
    let d_model = 8usize;
    let d_ff = 6usize;
    let silu_alpha = 1.702f32;
    let gate_clip = (-6.0f32, 7.0f32);
    let up_clip = (-6.0f32, 8.0f32);

    let x_seed = [0.25f32, -0.5, 1.0, -1.25, 0.75, -0.375, 0.125, 0.875];
    let x: Vec<bf16> = x_seed.iter().copied().map(bf16::from_f32).collect();
    let active_experts = [1usize, 3usize];
    let sum_k = active_experts.len();
    let k = active_experts.len();

    let mut x_perm = Vec::with_capacity(sum_k * d_model);
    for _ in &active_experts {
        x_perm.extend_from_slice(&x);
    }

    let mut expert_offsets = vec![0u32; e + 1];
    let mut running = 0u32;
    for expert in 0..e {
        expert_offsets[expert] = running;
        let count =
            active_experts.iter().filter(|&&idx| idx == expert).count() as u32;
        running += count;
    }
    expert_offsets[e] = running;

    let elem_w13 = e * d_model * 2 * d_ff;
    let elem_w2 = e * d_ff * d_model;
    let elem_up_bias = e * 2 * d_ff;
    let elem_down_bias = e * d_model;

    let mut base_w13 = vec![bf16::from_f32(0.0); elem_w13];
    for (expert, chunk) in base_w13.chunks_mut(d_model * 2 * d_ff).enumerate() {
        for dm in 0..d_model {
            for ch in 0..(2 * d_ff) {
                let idx = dm * 2 * d_ff + ch;
                let value = (expert as f32 + 1.0) * 0.05
                    + dm as f32 * 0.003
                    + ch as f32 * 0.001;
                chunk[idx] = bf16::from_f32(value);
            }
        }
    }

    let mut base_w2 = vec![bf16::from_f32(0.0); elem_w2];
    for (expert, chunk) in base_w2.chunks_mut(d_ff * d_model).enumerate() {
        for ff in 0..d_ff {
            for dm in 0..d_model {
                let idx = ff * d_model + dm;
                let value = (expert as f32 + 1.0) * 0.04
                    + ff as f32 * 0.002
                    + dm as f32 * 0.001;
                chunk[idx] = bf16::from_f32(value);
            }
        }
    }

    let mut base_up_bias = vec![bf16::from_f32(0.0); elem_up_bias];
    for (expert, chunk) in base_up_bias.chunks_mut(2 * d_ff).enumerate() {
        for ch in 0..(2 * d_ff) {
            let value = (expert as f32 + 1.0) * 0.03 + ch as f32 * 0.001;
            chunk[ch] = bf16::from_f32(value);
        }
    }

    let mut base_down_bias = vec![bf16::from_f32(0.0); elem_down_bias];
    for (expert, chunk) in base_down_bias.chunks_mut(d_model).enumerate() {
        for dm in 0..d_model {
            let value = (expert as f32 + 1.0) * 0.02 + dm as f32 * 0.001;
            chunk[dm] = bf16::from_f32(value);
        }
    }

    let fused_run = || -> Vec<f32> {
        let x_perm_buf = ctx.device.new_buffer_with_data(
            x_perm.as_ptr() as *const _,
            (x_perm.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let offsets_buf = ctx.device.new_buffer_with_data(
            expert_offsets.as_ptr() as *const _,
            (expert_offsets.len() * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w13_buf = ctx.device.new_buffer_with_data(
            base_w13.as_ptr() as *const _,
            (base_w13.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let w2_buf = ctx.device.new_buffer_with_data(
            base_w2.as_ptr() as *const _,
            (base_w2.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let up_bias_buf = ctx.device.new_buffer_with_data(
            base_up_bias.as_ptr() as *const _,
            (base_up_bias.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let down_bias_buf = ctx.device.new_buffer_with_data(
            base_down_bias.as_ptr() as *const _,
            (base_down_bias.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let y_partial_buf = ctx.device.new_buffer(
            (sum_k * d_model * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        const BN: usize = 64;
        let num_tiles_n = (d_model + BN - 1) / BN;
        let max_tiles = sum_k * e * num_tiles_n;
        let tile_counts_buf = ctx.device.new_buffer(
            (e * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tile_offsets_buf = ctx.device.new_buffer(
            ((e + 1) * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tile_map_buf = ctx.device.new_buffer(
            (max_tiles * 3 * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let total_tiles_buf = ctx.device.new_buffer(
            (8 * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dispatch_args_buf = ctx.device.new_buffer(
            (3 * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let experts_kernel =
            MoeExpertsKernel::new(ctx).expect("experts kernel");

        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode(
                &cb,
                MoeExpertsArguments {
                    x_perm_buffer: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    w13_all: &w13_buf,
                    w2_all: &w2_buf,
                    y_partial: &y_partial_buf,
                    up_biases: &up_bias_buf,
                    down_biases: &down_bias_buf,
                    tile_counts: &tile_counts_buf,
                    tile_row_offsets: &tile_offsets_buf,
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

        let slice = unsafe {
            std::slice::from_raw_parts(
                y_partial_buf.contents() as *const bf16,
                sum_k * d_model,
            )
        };
        slice.iter().map(|&v| f32::from(v)).collect()
    };

    let two_pass_run = || -> Vec<f32> {
        let x_perm_buf = ctx.device.new_buffer_with_data(
            x_perm.as_ptr() as *const _,
            (x_perm.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let offsets_buf = ctx.device.new_buffer_with_data(
            expert_offsets.as_ptr() as *const _,
            (expert_offsets.len() * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let w13_buf = ctx.device.new_buffer_with_data(
            base_w13.as_ptr() as *const _,
            (base_w13.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let w2_buf = ctx.device.new_buffer_with_data(
            base_w2.as_ptr() as *const _,
            (base_w2.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let up_bias_buf = ctx.device.new_buffer_with_data(
            base_up_bias.as_ptr() as *const _,
            (base_up_bias.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let down_bias_buf = ctx.device.new_buffer_with_data(
            base_down_bias.as_ptr() as *const _,
            (base_down_bias.len() * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

        let hidden_buf = ctx.device.new_buffer(
            (sum_k * d_ff * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let partial_buf = ctx.device.new_buffer(
            (num_tiles_k * sum_k * d_model * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = ctx.device.new_buffer(
            (sum_k * d_model * size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let experts_kernel =
            MoeExpertsKernel::new(ctx).expect("experts kernel");

        let args = MoeExpertsTwoPassArguments {
            x_perm_buffer: &x_perm_buf,
            expert_offsets: &offsets_buf,
            hidden_buffer: &hidden_buf,
            partial_buffer: &partial_buf,
            output_buffer: &output_buf,
            w13_all: &w13_buf,
            w2_all: &w2_buf,
            up_biases: &up_bias_buf,
            down_biases: &down_bias_buf,
            total_rows: sum_k,
            d_model,
            d_ff,
            e,
            num_tiles_k: num_tiles_k as u32,
            gating_code,
            gate_clip_min: gate_clip.0,
            gate_clip_max: gate_clip.1,
            up_clip_min: up_clip.0,
            up_clip_max: up_clip.1,
            silu_alpha,
            data_type: dtype,
        };

        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode_two_pass_decode(&cb, args)
            .expect("two-pass encode");
        cb.commit();
        cb.wait_until_completed();

        let slice = unsafe {
            std::slice::from_raw_parts(
                output_buf.contents() as *const bf16,
                sum_k * d_model,
            )
        };
        slice.iter().map(|&v| f32::from(v)).collect()
    };

    let y_gpu = if force_two_pass {
        two_pass_run()
    } else {
        fused_run()
    };

    let mut expected = Vec::with_capacity(sum_k * d_model);
    for (row_idx, &expert_idx) in active_experts.iter().enumerate() {
        let base = expert_idx;
        let w13_slice = &base_w13
            [base * d_model * 2 * d_ff..(base + 1) * d_model * 2 * d_ff];
        let w2_slice =
            &base_w2[base * d_ff * d_model..(base + 1) * d_ff * d_model];
        let up_bias_slice =
            &base_up_bias[base * 2 * d_ff..(base + 1) * 2 * d_ff];
        let down_bias_slice =
            &base_down_bias[base * d_model..(base + 1) * d_model];

        let y = cpu_expert_forward(
            &x_perm[row_idx * d_model..(row_idx + 1) * d_model],
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
        KernelDataType::BFloat16 => 3e-3f32,
        KernelDataType::Float32 => 1e-4f32,
        KernelDataType::Float16 => unreachable!(),
    };

    for i in 0..expected.len() {
        max_abs = max_abs.max((y_gpu[i] - expected[i]).abs());
    }

    assert!(
        max_abs < tol,
        "max diff {} exceeds tol {} (dtype {:?}, gating {}, two_pass={})",
        max_abs,
        tol,
        dtype,
        gating_code,
        force_two_pass,
    );

    max_abs
}

#[test]
fn test_moe_experts_decode_parity() {
    let ctx = create_ctx();
    for &dtype in &[KernelDataType::BFloat16] {
        for &force_two_pass in &[false, true] {
            for gating_code in 0u32..=3u32 {
                run_decode_parity_case(
                    &ctx,
                    dtype,
                    gating_code,
                    force_two_pass,
                );
            }
        }
    }
}
