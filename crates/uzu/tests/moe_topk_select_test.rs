#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::{MoeTopKArguments, MoeTopKError, MoeTopKKernel},
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn cpu_topk_select(
    logits: &[f32],
    t: usize,
    e: usize,
    k: usize,
    renorm: bool,
) -> (Vec<i32>, Vec<f32>) {
    assert_eq!(logits.len(), t * e);
    assert!(e >= k && k >= 1);

    let mut ids = vec![0i32; t * k];
    let mut probs = vec![0f32; t * k];
    for ti in 0..t {
        let row = &logits[ti * e..(ti + 1) * e];
        let mut best_vals = vec![f32::NEG_INFINITY; k];
        let mut best_ids = vec![-1i32; k];
        for eid in 0..e {
            let v = row[eid];
            let mut insert_pos: Option<usize> = None;
            for j in (0..k).rev() {
                if v > best_vals[j]
                    || (v == best_vals[j] && (eid as i32) < best_ids[j])
                {
                    insert_pos = Some(j);
                }
            }
            if let Some(pos) = insert_pos {
                for s in (pos + 1..k).rev() {
                    best_vals[s] = best_vals[s - 1];
                    best_ids[s] = best_ids[s - 1];
                }
                best_vals[pos] = v;
                best_ids[pos] = eid as i32;
            }
        }
        let base = ti * k;
        for kk in 0..k {
            ids[base + kk] = best_ids[kk];
        }
        if renorm {
            let max_v =
                best_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut exps = vec![0.0f32; k];
            for kk in 0..k {
                exps[kk] = (best_vals[kk] - max_v).exp();
                sum += exps[kk];
            }
            for kk in 0..k {
                probs[base + kk] = if sum > 0.0 {
                    exps[kk] / sum
                } else {
                    1.0 / (k as f32)
                };
            }
        } else {
            for kk in 0..k {
                probs[base + kk] = best_vals[kk];
            }
        }
    }
    (ids, probs)
}

fn run_topk_once(
    ctx: &MTLContext,
    dtype: KernelDataType,
    logits_f32: &[f32],
    t: usize,
    e: usize,
    k: usize,
    renorm: bool,
) -> Result<(Vec<i32>, Vec<f32>), MoeTopKError> {
    let kernel = MoeTopKKernel::new(ctx)?;

    let (logits_buf, probs_elem_count) = match dtype {
        KernelDataType::Float32 => {
            let buf = ctx.device.new_buffer_with_data(
                logits_f32.as_ptr() as *const _,
                (logits_f32.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf, t * k)
        },
        KernelDataType::Float16 => {
            let logits_f16: Vec<f16> =
                logits_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let buf = ctx.device.new_buffer_with_data(
                logits_f16.as_ptr() as *const _,
                (logits_f16.len() * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf, t * k)
        },
        KernelDataType::BFloat16 => {
            unreachable!("Not supported in v1");
        },
    };

    let ids_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let probs_buf = ctx.device.new_buffer(
        (probs_elem_count * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    let args = MoeTopKArguments {
        logits_buffer: &logits_buf,
        topk_ids_buffer: &ids_buf,
        topk_probs_buffer: &probs_buf,
        t,
        e,
        k,
        renorm,
    };

    kernel.encode(&enc, dtype, args)?;

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let ids_ptr = ids_buf.contents() as *const i32;
    let probs_ptr = probs_buf.contents() as *const f16;
    let ids_slice = unsafe { std::slice::from_raw_parts(ids_ptr, t * k) };
    let probs_slice = unsafe { std::slice::from_raw_parts(probs_ptr, t * k) };
    let probs_f32: Vec<f32> = probs_slice.iter().map(|&h| h.to_f32()).collect();

    Ok((ids_slice.to_vec(), probs_f32))
}

#[test]
fn test_topk_correctness_random() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping MoE topk test");
            return;
        },
    };

    let mut rng = StdRng::seed_from_u64(42);
    let shapes = vec![(1usize, 4usize), (4, 16), (32, 64), (32, 257)];
    let ks = vec![1usize, 2usize];
    let renorms = vec![false, true];
    for &(t, e) in &shapes {
        for &k in &ks {
            if e < k {
                continue;
            }
            for &renorm in &renorms {
                let logits: Vec<f32> =
                    (0..t * e).map(|_| rng.random_range(-5.0..5.0)).collect();

                let (ref_ids, ref_probs) =
                    cpu_topk_select(&logits, t, e, k, renorm);
                let (gpu_ids, gpu_probs) = run_topk_once(
                    &ctx,
                    KernelDataType::Float32,
                    &logits,
                    t,
                    e,
                    k,
                    renorm,
                )
                .expect("encode/run");

                assert_eq!(
                    gpu_ids, ref_ids,
                    "ids mismatch (T={},E={},K={},renorm={})",
                    t, e, k, renorm
                );
                let atol = if renorm {
                    1e-3
                } else {
                    2e-3
                };
                for i in 0..(t * k) {
                    let a = gpu_probs[i];
                    let b = ref_probs[i];
                    let diff = (a - b).abs();
                    assert!(
                        diff <= atol,
                        "probs mismatch at {}: got {} ref {} (|diff|={} > atol={}), T={} E={} K={} renorm={}",
                        i,
                        a,
                        b,
                        diff,
                        atol,
                        t,
                        e,
                        k,
                        renorm
                    );
                }
            }
        }
    }
}

#[test]
fn test_topk_ties_and_ordering() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping MoE topk ties test");
            return;
        },
    };

    // T=2, E=5, K=2 with crafted ties
    let t = 2usize;
    let e = 5usize;
    let k = 2usize;
    let renorm = true;
    let mut logits = vec![0.0f32; t * e];
    // row 0: equal top at e=1 and e=2
    logits[0 * e + 1] = 3.0;
    logits[0 * e + 2] = 3.0; // tie -> prefer smaller id (1)
    logits[0 * e + 3] = 2.9;
    // row 1: strictly descending
    logits[1 * e + 4] = 5.0;
    logits[1 * e + 0] = 4.0;

    let (ref_ids, _ref_probs) = cpu_topk_select(&logits, t, e, k, renorm);
    let (gpu_ids, _gpu_probs) =
        run_topk_once(&ctx, KernelDataType::Float32, &logits, t, e, k, renorm)
            .expect("encode/run");

    assert_eq!(gpu_ids, ref_ids);

    // Ordering property: ids correspond to non-increasing logits with our tie rule
    for row in 0..t {
        let base = row * k;
        let id0 = gpu_ids[base + 0] as usize;
        if k > 1 {
            let id1 = gpu_ids[base + 1] as usize;
            let v0 = logits[row * e + id0];
            let v1 = logits[row * e + id1];
            assert!(v0 > v1 || (v0 == v1 && id0 < id1));
        }
    }
}

#[test]
fn test_topk_edge_cases_and_determinism() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping MoE topk edge test");
            return;
        },
    };

    // T==0 no-op
    {
        let t = 0usize;
        let e = 8usize;
        let k = 2usize;
        let logits: Vec<f32> = vec![];
        let res = run_topk_once(
            &ctx,
            KernelDataType::Float32,
            &logits,
            t,
            e,
            k,
            false,
        );
        assert!(res.is_ok());
        let (ids, probs) = res.unwrap();
        assert!(ids.is_empty() && probs.is_empty());
    }

    // E==K valid
    {
        let t = 4usize;
        let e = 2usize;
        let k = 2usize;
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.0, -1.0, 2.5, 2.5];
        let (ids, _probs) = run_topk_once(
            &ctx,
            KernelDataType::Float32,
            &logits,
            t,
            e,
            k,
            true,
        )
        .expect("encode/run");
        assert_eq!(ids.len(), t * k);
    }

    // E < K -> error
    {
        let t = 1usize;
        let e = 1usize;
        let k = 2usize;
        let logits: Vec<f32> = vec![0.0];
        let kernel = MoeTopKKernel::new(&ctx).expect("kernel");
        let logits_buf = ctx.device.new_buffer_with_data(
            logits.as_ptr() as *const _,
            (logits.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ids_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let probs_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let args = MoeTopKArguments {
            logits_buffer: &logits_buf,
            topk_ids_buffer: &ids_buf,
            topk_probs_buffer: &probs_buf,
            t,
            e,
            k,
            renorm: true,
        };
        let err =
            kernel.encode(&enc, KernelDataType::Float32, args).unwrap_err();
        match err {
            MoeTopKError::InvalidDimensions {
                ..
            } => {},
            other => panic!("Unexpected error: {:?}", other),
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Determinism: run twice -> identical bytes
    {
        let t = 16usize;
        let e = 33usize;
        let k = 2usize;
        let mut rng = StdRng::seed_from_u64(123);
        let logits: Vec<f32> =
            (0..t * e).map(|_| rng.random_range(-3.0..3.0)).collect();
        let (ids1, probs1) = run_topk_once(
            &ctx,
            KernelDataType::Float32,
            &logits,
            t,
            e,
            k,
            true,
        )
        .expect("run1");
        let (ids2, probs2) = run_topk_once(
            &ctx,
            KernelDataType::Float32,
            &logits,
            t,
            e,
            k,
            true,
        )
        .expect("run2");
        assert_eq!(ids1, ids2);
        assert_eq!(probs1, probs2);
    }
}
