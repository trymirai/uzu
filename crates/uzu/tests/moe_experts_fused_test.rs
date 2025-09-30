#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use uzu::backends::metal::{
    MTLContext,
    kernel::{MoeExpertsArguments, MoeExpertsKernel},
};

fn create_ctx() -> MTLContext {
    let d = Device::system_default().expect("no metal");
    let q = d.new_command_queue();
    MTLContext::new(d, q).expect("ctx")
}

fn cpu_ref(
    x: &[f16],
    _t: usize,
    d_model: usize,
    bucket_ids: &[i32],
    offsets: &[u32],
    e: usize,
    w1: &[f16],
    w3: Option<&[f16]>,
    w2: &[f16],
    d_ff: usize,
    gating: u32,
) -> Vec<f16> {
    let sum_k = offsets[e] as usize;
    let mut y = vec![f16::from_f32(0.0); sum_k * d_model];
    let gelu = |v: f32| {
        0.5 * v
            * (1.0
                + (f32::sqrt(2.0 / std::f32::consts::PI)
                    * (v + 0.044715 * v * v * v))
                    .tanh())
    };
    let silu = |v: f32| v / (1.0 + (-v).exp());
    for i in 0..sum_k {
        let token = bucket_ids[i] as usize;
        let e_idx = (0..e)
            .find(|&ee| {
                (offsets[ee] as usize) <= i && i < (offsets[ee + 1] as usize)
            })
            .unwrap();
        let w1b = e_idx * d_ff * d_model;
        let w3b = e_idx * d_ff * d_model;
        let w2b = e_idx * d_model * d_ff;
        // up/gate
        let mut a = vec![0f32; d_ff];
        for r in 0..d_ff {
            let mut up = 0f32;
            let mut vp = 0f32;
            for c in 0..d_model {
                let xij = x[token * d_model + c].to_f32();
                up += w1[w1b + r * d_model + c].to_f32() * xij;
                if let Some(w3s) = w3 {
                    vp += w3s[w3b + r * d_model + c].to_f32() * xij;
                }
            }
            a[r] = match gating {
                0 => gelu(up),
                1 => silu(up),
                2 => { let s = silu(vp); s * up + s },  // OpenAI: silu(gate)*up + silu(gate)
                _ => { let g = gelu(vp); g * up + g },
            };
        }
        for c in 0..d_model {
            let mut acc = 0f32;
            for r in 0..d_ff {
                acc += w2[w2b + c * d_ff + r].to_f32() * a[r];
            }
            y[i * d_model + c] = f16::from_f32(acc);
        }
    }
    y
}

#[test]
fn test_moe_fused_expert_mlp_parity() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(2025);
    let t = 37usize;
    let e = 9usize;
    let d_model = 64usize;
    let d_ff = 128usize;
    let k = 2usize;
    // Inputs
    let x: Vec<f16> = (0..t * d_model)
        .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    let bucket_ids: Vec<i32> =
        (0..t * k).map(|i| (i as i32) % t as i32).collect();
    // simple offsets: each expert gets ~equal tokens
    let sum_k = t * k;
    let mut offsets = vec![0u32; e + 1];
    for i in 0..e {
        offsets[i + 1] = (((i + 1) * sum_k) / e) as u32;
    }
    // Weights
    let w1: Vec<f16> = (0..e * d_ff * d_model)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let w3: Vec<f16> = (0..e * d_ff * d_model)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let w2: Vec<f16> = (0..e * d_model * d_ff)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bid_buf = ctx.device.new_buffer_with_data(
        bucket_ids.as_ptr() as *const _,
        (bucket_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let off_buf = ctx.device.new_buffer_with_data(
        offsets.as_ptr() as *const _,
        (offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w1_buf = ctx.device.new_buffer_with_data(
        w1.as_ptr() as *const _,
        (w1.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w3_buf = ctx.device.new_buffer_with_data(
        w3.as_ptr() as *const _,
        (w3.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w2_buf = ctx.device.new_buffer_with_data(
        w2.as_ptr() as *const _,
        (w2.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut y = vec![f16::from_f32(0.0); sum_k * d_model];
    let y_buf = ctx.device.new_buffer_with_data(
        y.as_ptr() as *const _,
        (y.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kf = MoeExpertsKernel::new(&ctx).expect("experts");
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    
    let up_bias_size = (e * d_ff * 2 * std::mem::size_of::<f16>()) as u64;
    let down_bias_size = (e * d_model * std::mem::size_of::<f16>()) as u64;
    let up_biases_buf = ctx.device.new_buffer(up_bias_size, metal::MTLResourceOptions::StorageModeShared);
    let down_biases_buf = ctx.device.new_buffer(down_bias_size, metal::MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::write_bytes(up_biases_buf.contents() as *mut u8, 0, up_bias_size as usize);
        std::ptr::write_bytes(down_biases_buf.contents() as *mut u8, 0, down_bias_size as usize);
    }
    
    kf.encode(
        &enc,
        MoeExpertsArguments {
            x_buffer: &x_buf,
            bucketed_token_ids: &bid_buf,
            expert_offsets: &off_buf,
            w1_all: &w1_buf,
            w3_all: &w3_buf,
            w2_all: &w2_buf,
            y_partial: &y_buf,
            up_biases: &up_biases_buf,
            down_biases: &down_biases_buf,
            t,
            d_model,
            d_ff,
            e,
            gating_code: 2,
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            silu_alpha: 1.0,
        },
    )
    .expect("encode experts");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let y_gpu = unsafe {
        std::slice::from_raw_parts(
            y_buf.contents() as *const f16,
            sum_k * d_model,
        )
    };
    // Clean test: no debug prints
    let y_cpu = cpu_ref(
        &x,
        t,
        d_model,
        &bucket_ids,
        &offsets,
        e,
        &w1,
        Some(&w3),
        &w2,
        d_ff,
        2,
    );
    for i in 0..(sum_k * d_model) {
        let a = y_gpu[i].to_f32();
        let b = y_cpu[i].to_f32();
        assert!(
            (a - b).abs() < 1e-2,
            "mismatch at {}: {} vs {} (diff={})",
            i,
            a,
            b,
            (a - b).abs()
        );
    }
}

#[test]
fn test_moe_fused_expert_mlp_large_dmodel_tail_dff_and_non_glu() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(42);
    let t = 13usize;
    let e = 5usize;
    let d_model = 384usize;
    let d_ff = 160usize;
    let k = 1usize;
    // Inputs
    let x: Vec<f16> = (0..t * d_model)
        .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    // Build offsets with empty and skewed segments
    // Expert 0: 0 tokens, 1: 2 tokens, 2: 7 tokens, 3: 0 tokens, 4: 4 tokens -> total 13
    let offsets: Vec<u32> = vec![0, 0, 2, 9, 9, 13];
    // Randomized bucket ids consistent with offsets
    let mut pool: Vec<i32> = (0..t as i32).collect();
    pool.shuffle(&mut rng);
    let bucket_ids: Vec<i32> = pool;

    // Weights
    let w1: Vec<f16> = (0..e * d_ff * d_model)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let w2: Vec<f16> = (0..e * d_model * d_ff)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bid_buf = ctx.device.new_buffer_with_data(
        bucket_ids.as_ptr() as *const _,
        (bucket_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let off_buf = ctx.device.new_buffer_with_data(
        offsets.as_ptr() as *const _,
        (offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w1_buf = ctx.device.new_buffer_with_data(
        w1.as_ptr() as *const _,
        (w1.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w3_dummy: Vec<f16> = vec![f16::from_f32(0.0); e * d_ff * d_model];
    let w3_buf = ctx.device.new_buffer_with_data(
        w3_dummy.as_ptr() as *const _,
        (w3_dummy.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w2_buf = ctx.device.new_buffer_with_data(
        w2.as_ptr() as *const _,
        (w2.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut y = vec![f16::from_f32(0.0); t * d_model];
    let y_buf = ctx.device.new_buffer_with_data(
        y.as_ptr() as *const _,
        (y.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kf = MoeExpertsKernel::new(&ctx).expect("experts");
    for &gate in &[0u32, 1u32] {
        // GELU and SiLU
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        
        let up_bias_size = (e * d_ff * 2 * std::mem::size_of::<f16>()) as u64;
        let down_bias_size = (e * d_model * std::mem::size_of::<f16>()) as u64;
        let up_biases_buf = ctx.device.new_buffer(up_bias_size, metal::MTLResourceOptions::StorageModeShared);
        let down_biases_buf = ctx.device.new_buffer(down_bias_size, metal::MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(up_biases_buf.contents() as *mut u8, 0, up_bias_size as usize);
            std::ptr::write_bytes(down_biases_buf.contents() as *mut u8, 0, down_bias_size as usize);
        }
        
        kf.encode(
            &enc,
            MoeExpertsArguments {
                x_buffer: &x_buf,
                bucketed_token_ids: &bid_buf,
                expert_offsets: &off_buf,
                w1_all: &w1_buf,
                w3_all: &w3_buf,
                w2_all: &w2_buf,
                y_partial: &y_buf,
                up_biases: &up_biases_buf,
                down_biases: &down_biases_buf,
                t,
                d_model,
                d_ff,
                e,
                gating_code: gate,
                gate_clip_min: f32::NEG_INFINITY,
                gate_clip_max: f32::INFINITY,
                up_clip_min: f32::NEG_INFINITY,
                up_clip_max: f32::INFINITY,
                silu_alpha: 1.0,
            },
        )
        .expect("encode experts");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let y_gpu = unsafe {
            std::slice::from_raw_parts(
                y_buf.contents() as *const f16,
                t * d_model,
            )
        };
        // Clean test: no debug prints
        let y_cpu = cpu_ref(
            &x,
            t,
            d_model,
            &bucket_ids,
            &offsets,
            e,
            &w1,
            None,
            &w2,
            d_ff,
            gate,
        );

        // Clean test: no float-out or extra checks

        for i in 0..(t * d_model) {
            let a = y_gpu[i].to_f32();
            let b = y_cpu[i].to_f32();
            assert!(
                (a - b).abs() < 1e-2,
                "gate {} mismatch at {}: {} vs {} (diff={})",
                gate,
                i,
                a,
                b,
                (a - b).abs()
            );
        }
    }
}
