use std::{
    collections::HashSet,
    ops::{Deref, DerefMut},
};

use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::common::{
        Backend, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferInitial, CommandBufferPending, Context,
        kernel::sampling::SamplingKernel,
    },
    session::parameter::SamplingMethod,
};

const SEED: u64 = 42;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn logits_from_probs(probs: &[f32]) -> Vec<f32> {
    probs
        .iter()
        .map(|&p| if p <= 0.0 { -50.0 } else { p.ln() })
        .collect()
}

fn run_samples<B: Backend>(
    context: &B::Context,
    kernel: &SamplingKernel<B>,
    logits: &[f32],
    batch_size: usize,
    vocab_size: usize,
    method: SamplingMethod,
    num_samples: usize,
) -> Vec<Vec<u32>> {
    let output_array = context.create_array_uninitialized(&[batch_size], DataType::U32, "");
    let mut per_batch_samples: Vec<Vec<u32>> = vec![Vec::new(); batch_size];

    for draw in 0..num_samples {
        let logits_array = context.create_array_from(&[batch_size * vocab_size], logits, "");
        let seeds: Vec<u64> = vec![SEED + draw as u64; batch_size];
        let seeds_array = context.create_array_from(&[batch_size], &seeds, "");

        let mut command_buffer =
            context.create_command_buffer().expect("create command buffer").start_encoding();
        kernel
            .encode(
                logits_array.buffer().borrow_mut().deref_mut(),
                Some(seeds_array.buffer().borrow().deref()),
                0,
                None,
                0,
                output_array.buffer().borrow_mut().deref_mut(),
                method,
                batch_size,
                vocab_size,
                &mut command_buffer,
            )
            .expect("encode");
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();

        let sampled: &[u32] = output_array.as_slice();
        for (b, &tok) in sampled.iter().enumerate() {
            per_batch_samples[b].push(tok);
        }
    }

    per_batch_samples
}

// ── Constraint satisfaction ───────────────────────────────────────────────────
//
// Builds a distribution where k tokens have equal high probability (p=0.1 each)
// and the remaining vocab_size - k tokens share the remaining mass equally.
// With top_p = k * 0.1, only the k high-probability tokens should ever be sampled.

fn test_constraint_satisfaction<B: Backend>(
    batch_size: usize,
    k: usize,
    vocab_size: usize,
    method: SamplingMethod,
) {
    let context = <B as Backend>::Context::new().expect("create context");
    let kernel = SamplingKernel::<B>::new(&context, DataType::F32, batch_size, vocab_size)
        .expect("create kernel");

    let low_prob = (1.0 - k as f32 * 0.1) / (vocab_size - k) as f32;
    let mut probs = vec![low_prob; batch_size * vocab_size];
    let mut high_prob_sets: Vec<HashSet<usize>> = Vec::with_capacity(batch_size);

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut token_ids: Vec<usize> = (0..vocab_size).collect();

    for b in 0..batch_size {
        token_ids.shuffle(&mut rng);
        let high: HashSet<usize> = token_ids[..k].iter().copied().collect();
        for &id in &high {
            probs[b * vocab_size + id] = 0.1;
        }
        high_prob_sets.push(high);
    }

    let logits = logits_from_probs(&probs);
    let samples = run_samples::<B>(&context, &kernel, &logits, batch_size, vocab_size, method, 500);

    for (b, batch_samples) in samples.iter().enumerate() {
        for &tok in batch_samples {
            assert!(
                high_prob_sets[b].contains(&(tok as usize)),
                "batch {b}: sampled token {tok} not in high-probability set (method={method:?})"
            );
        }
    }
}

#[test]
fn test_top_p_only_never_samples_outside_support() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: None,
                top_p: Some(10.0 * 0.1),
                min_p: None,
            },
        );
    });
}

#[test]
fn test_top_k_only_never_samples_outside_support() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: Some(10),
                top_p: None,
                min_p: None,
            },
        );
    });
}

#[test]
fn test_top_k_and_top_p_combined() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: Some(20),    // wider than top_p, so top_p is the binding constraint
                top_p: Some(10.0 * 0.1),
                min_p: None,
            },
        );
    });
}

#[test]
fn test_temperature_with_top_p() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::Stochastic {
                temperature: Some(0.5),
                top_k: None,
                top_p: Some(10.0 * 0.1),
                min_p: None,
            },
        );
    });
}

#[test]
fn test_all_params_combined() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::Stochastic {
                temperature: Some(0.8),
                top_k: Some(20),
                top_p: Some(10.0 * 0.1),
                min_p: None,
            },
        );
    });
}

// ── Statistical correctness (KL divergence vs CPU reference) ─────────────────

fn cpu_reference_top_p(logits: &[f32], top_p: f32) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum = 0.0f32;
    let mut keep = vec![false; logits.len()];
    for &(idx, p) in &indexed {
        if cum >= top_p {
            break;
        }
        keep[idx] = true;
        cum += p;
    }

    let kept_sum: f32 = probs.iter().zip(&keep).filter(|&(_, &k)| k).map(|(&p, _)| p).sum();
    probs
        .iter()
        .zip(&keep)
        .map(|(&p, &k)| if k { p / kept_sum } else { 0.0 })
        .collect()
}

#[test]
fn test_statistical_correctness_top_p() {
    for_each_non_cpu_backend!(|B| {
        const BATCH: usize = 16;
        const VOCAB: usize = 2048;
        const NUM_DRAWS: usize = 5_000;
        const TOP_P: f32 = 0.9;
        const TOLERANCE_KL: f32 = 0.05;

        let context = <B as Backend>::Context::new().expect("create context");
        let kernel =
            SamplingKernel::<B>::new(&context, DataType::F32, BATCH, VOCAB).expect("create kernel");

        let mut rng = StdRng::seed_from_u64(SEED);
        let mut logits = vec![0.0f32; BATCH * VOCAB];
        for x in logits.iter_mut() {
            *x = rng.random_range(-6.0f32..6.0f32);
        }

        // Build reference distribution per row.
        let mut ref_probs = vec![0.0f32; BATCH * VOCAB];
        for b in 0..BATCH {
            let row = &logits[b * VOCAB..(b + 1) * VOCAB];
            let dist = cpu_reference_top_p(row, TOP_P);
            ref_probs[b * VOCAB..(b + 1) * VOCAB].copy_from_slice(&dist);
        }

        let output_array = context.create_array_uninitialized(&[BATCH], DataType::U32, "");
        let mut counters = vec![0u32; BATCH * VOCAB];

        for draw in 0..NUM_DRAWS {
            let logits_array = context.create_array_from(&[BATCH * VOCAB], &logits, "");
            let seeds: Vec<u64> = vec![SEED + draw as u64; BATCH];
            let seeds_array = context.create_array_from(&[BATCH], &seeds, "");

            let mut command_buffer =
                context.create_command_buffer().expect("create command buffer").start_encoding();
            kernel
                .encode(
                    logits_array.buffer().borrow_mut().deref_mut(),
                    Some(seeds_array.buffer().borrow().deref()),
                    0,
                    None,
                    0,
                    output_array.buffer().borrow_mut().deref_mut(),
                    SamplingMethod::Stochastic {
                        temperature: None,
                        top_k: None,
                        top_p: Some(TOP_P),
                        min_p: None,
                    },
                    BATCH,
                    VOCAB,
                    &mut command_buffer,
                )
                .expect("encode");
            command_buffer.end_encoding().submit().wait_until_completed().unwrap();

            let sample_ids: &[u32] = output_array.as_slice();
            for (b, &tok) in sample_ids.iter().enumerate() {
                counters[b * VOCAB + tok as usize] += 1;
            }
        }

        for b in 0..BATCH {
            let mut kl = 0.0f32;
            for j in 0..VOCAB {
                let expected = ref_probs[b * VOCAB + j];
                if expected < 1e-12 {
                    continue;
                }
                let observed = (counters[b * VOCAB + j] as f32) / (NUM_DRAWS as f32);
                if observed > 0.0 {
                    kl += observed * (observed.ln() - expected.ln());
                }
            }
            assert!(
                kl < TOLERANCE_KL,
                "batch {b}: KL divergence {kl:.4} exceeds tolerance {TOLERANCE_KL:.3} (backend: {})",
                std::any::type_name::<B>()
            );
        }
    });
}

// ── Constraint tests for UnifiedStochastic ────────────────────────────────

#[test]
fn test_single_pass_top_p_only() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::UnifiedStochastic {
                temperature: None,
                top_k: None,
                top_p: Some(10.0 * 0.1),
                min_p: None,
            },
        );
    });
}

#[test]
fn test_single_pass_top_k_only() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::UnifiedStochastic {
                temperature: None,
                top_k: Some(10),
                top_p: None,
                min_p: None,
            },
        );
    });
}

#[test]
fn test_single_pass_all_params() {
    for_each_non_cpu_backend!(|B| {
        test_constraint_satisfaction::<B>(
            8,
            10,
            1024,
            SamplingMethod::UnifiedStochastic {
                temperature: Some(0.8),
                top_k: Some(20),
                top_p: Some(10.0 * 0.1),
                min_p: None,
            },
        );
    });
}

// ── Performance baseline ──────────────────────────────────────────────────────
//
// Measures GPU execution time for the current sequential multi-kernel sampling
// path. Run with: cargo test perf_ -- --nocapture

fn time_encode<B: Backend, F>(
    context: &B::Context,
    logits: &[f32],
    seeds: &[u64],
    batch_size: usize,
    vocab_size: usize,
    warmup: usize,
    runs: usize,
    mut encode_fn: F,
) -> Vec<f64>
where
    F: FnMut(
        &mut B::Buffer,
        &B::Buffer,
        &mut B::Buffer,
        &mut <B::CommandBuffer as uzu::backends::common::CommandBuffer>::Encoding,
    ),
{
    for _ in 0..warmup {
        let logits_array = context.create_array_from(&[batch_size * vocab_size], logits, "");
        let seeds_array = context.create_array_from(&[batch_size], seeds, "");
        let output_array = context.create_array_uninitialized(&[batch_size], DataType::U32, "");
        let mut cb = context.create_command_buffer().unwrap().start_encoding();
        encode_fn(
            logits_array.buffer().borrow_mut().deref_mut(),
            seeds_array.buffer().borrow().deref(),
            output_array.buffer().borrow_mut().deref_mut(),
            &mut cb,
        );
        cb.end_encoding().submit().wait_until_completed().unwrap();
    }

    let mut times_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let logits_array = context.create_array_from(&[batch_size * vocab_size], logits, "");
        let seeds_array = context.create_array_from(&[batch_size], seeds, "");
        let output_array = context.create_array_uninitialized(&[batch_size], DataType::U32, "");
        let mut cb = context.create_command_buffer().unwrap().start_encoding();
        encode_fn(
            logits_array.buffer().borrow_mut().deref_mut(),
            seeds_array.buffer().borrow().deref(),
            output_array.buffer().borrow_mut().deref_mut(),
            &mut cb,
        );
        let completed = cb.end_encoding().submit().wait_until_completed().unwrap();
        if let Some(t) = completed.gpu_execution_time() {
            times_ms.push(t.as_secs_f64() * 1e3);
        }
    }
    times_ms
}

#[test]
fn perf_sequential_baseline_128k_vocab() {
    for_each_non_cpu_backend!(|B| {
        const BATCH: usize = 8;
        const VOCAB: usize = 128_000;
        const WARMUP: usize = 5;
        const RUNS: usize = 50;

        let context = <B as Backend>::Context::new().expect("create context");
        let kernel =
            SamplingKernel::<B>::new(&context, DataType::F32, BATCH, VOCAB).expect("create kernel");

        let mut rng = StdRng::seed_from_u64(SEED);
        let mut logits = vec![0.0f32; BATCH * VOCAB];
        for x in logits.iter_mut() {
            *x = rng.random_range(-6.0f32..6.0f32);
        }
        let seeds: Vec<u64> = vec![SEED; BATCH];

        // Matches the real-world default from GenerationConfig: temperature=0.6, top_k=20, top_p=0.95.
        let method = SamplingMethod::Stochastic {
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: Some(0.95),
            min_p: None,
        };

        let times = time_encode::<B, _>(
            &context,
            &logits,
            &seeds,
            BATCH,
            VOCAB,
            WARMUP,
            RUNS,
            |logits_buf, seeds_buf, out_buf, cb| {
                kernel
                    .encode(logits_buf, Some(seeds_buf), 0, None, 0, out_buf, method, BATCH, VOCAB, cb)
                    .unwrap();
            },
        );

        let single_pass_times = time_encode::<B, _>(
            &context,
            &logits,
            &seeds,
            BATCH,
            VOCAB,
            WARMUP,
            RUNS,
            |logits_buf, seeds_buf, out_buf, cb| {
                kernel
                    .encode(
                        logits_buf,
                        Some(seeds_buf),
                        0,
                        None,
                        0,
                        out_buf,
                        SamplingMethod::UnifiedStochastic {
                            temperature: Some(0.6),
                            top_k: Some(20),
                            top_p: Some(0.95),
                            min_p: None,
                        },
                        BATCH,
                        VOCAB,
                        cb,
                    )
                    .unwrap();
            },
        );

        let summarize = |times: &[f64]| -> String {
            if times.is_empty() {
                return "(no GPU timing available)".to_string();
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            format!("mean={mean:.3}ms  min={min:.3}ms")
        };

        println!(
            "[{}] batch={BATCH} vocab={VOCAB}",
            std::any::type_name::<B>()
        );
        println!("  sequential:   {}", summarize(&times));
        println!("  single-pass:  {}", summarize(&single_pass_times));
    });
}

fn perf_compare<B: Backend>(batch: usize, vocab: usize) {
    const WARMUP: usize = 10;
    const RUNS: usize = 100;

    let context = <B as Backend>::Context::new().expect("create context");
    let kernel = SamplingKernel::<B>::new(&context, DataType::F32, batch, vocab).expect("create kernel");

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut logits = vec![0.0f32; batch * vocab];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }
    let seeds: Vec<u64> = vec![SEED; batch];

    let seq = time_encode::<B, _>(&context, &logits, &seeds, batch, vocab, WARMUP, RUNS, |lb, sb, ob, cb| {
        kernel
            .encode(
                lb,
                Some(sb),
                0,
                None,
                0,
                ob,
                SamplingMethod::Stochastic { temperature: Some(0.6), top_k: Some(20), top_p: Some(0.95), min_p: None },
                batch,
                vocab,
                cb,
            )
            .unwrap();
    });
    let sp = time_encode::<B, _>(&context, &logits, &seeds, batch, vocab, WARMUP, RUNS, |lb, sb, ob, cb| {
        kernel
            .encode(
                lb,
                Some(sb),
                0,
                None,
                0,
                ob,
                SamplingMethod::UnifiedStochastic {
                    temperature: Some(0.6),
                    top_k: Some(20),
                    top_p: Some(0.95),
                    min_p: None,
                },
                batch,
                vocab,
                cb,
            )
            .unwrap();
    });

    let summarize = |times: &[f64]| {
        if times.is_empty() {
            return "(no GPU timing)".to_string();
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        format!("mean={mean:.3}ms  min={min:.3}ms")
    };

    println!("[{}] batch={batch} vocab={vocab}", std::any::type_name::<B>());
    println!("  sequential: {}", summarize(&seq));
    println!("  unified:    {}", summarize(&sp));
    if !seq.is_empty() && !sp.is_empty() {
        let speedup = seq.iter().cloned().fold(f64::INFINITY, f64::min)
            / sp.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  speedup:      {speedup:.2}x");
    }
}

#[test]
fn perf_batch1_128k_vocab() {
    for_each_non_cpu_backend!(|B| { perf_compare::<B>(1, 128_000); });
}

#[test]
fn perf_batch64_128k_vocab() {
    for_each_non_cpu_backend!(|B| { perf_compare::<B>(64, 128_000); });
}
