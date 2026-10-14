use serde_json::{Value, json};
use uzu_engine::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            kernel::{ContextRingUpdateKernel, RepetitionPenaltyKernel, UnifiedSamplingKernel},
        },
        webgpu::WebGpu,
    },
    data_type::DataType,
};

type GpuContext = <WebGpu as Backend>::Context;
type SamplingKernel = <<WebGpu as Backend>::Kernels as Kernels>::UnifiedSamplingKernel;
type PenaltyKernel = <<WebGpu as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel;
type RingKernel = <<WebGpu as Backend>::Kernels as Kernels>::ContextRingUpdateKernel;

#[derive(Clone, Copy, Default)]
struct Policy {
    stochastic: bool,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
}

fn sample(
    context: &GpuContext,
    logits: &[f32],
    seeds: &[u64],
    mask: Option<&[u32]>,
    vocab: u32,
    dtype: DataType,
    policy: Policy,
) -> Result<Vec<u32>, String> {
    let batch = logits.len() as u32 / vocab;
    let mut logits_buffer = context.create_buffer(logits.len() * dtype.size_in_bytes()).map_err(|e| e.to_string())?;
    if dtype == DataType::BF16 {
        logits_buffer.copyin(&logits.iter().map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>());
    } else {
        logits_buffer.copyin(logits);
    }
    let mut seeds_buffer = context.create_buffer(seeds.len() * 8).map_err(|e| e.to_string())?;
    seeds_buffer.copyin(seeds);
    let mask_buffer = mask
        .map(|mask| {
            let mut buffer = context.create_buffer(mask.len() * 4).map_err(|e| e.to_string())?;
            buffer.copyin(mask);
            Ok::<_, String>(buffer)
        })
        .transpose()?;
    let mut output = context.create_buffer(batch as usize * 4).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    let kernel = SamplingKernel::new(
        context,
        dtype,
        policy.stochastic,
        mask.is_some(),
        policy.temperature.is_some(),
        policy.top_k.is_some(),
        policy.top_p.is_some(),
        policy.min_p.is_some(),
    )
    .map_err(|e| e.to_string())?;
    kernel.encode(
        &logits_buffer,
        &mut output,
        policy.stochastic.then_some(&seeds_buffer),
        mask_buffer.as_ref(),
        policy.temperature,
        policy.top_k,
        policy.top_p,
        policy.min_p,
        vocab,
        batch,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    Ok(output.copyout::<u32>())
}

fn gumbel(
    seed: u64,
    index: u32,
    vocab: u32,
) -> f32 {
    let block = index / 1024;
    let offset = vocab.div_ceil(4096) * (index % 1024) + block / 4;
    let mut counter = [offset, 0, 0, 0];
    let mut key = [seed as u32, (seed >> 32) as u32];
    for _ in 0..10 {
        let first = 0xd2511f53u64 * u64::from(counter[0]);
        let second = 0xcd9e8d57u64 * u64::from(counter[2]);
        counter = [
            (second >> 32) as u32 ^ counter[1] ^ key[0],
            second as u32,
            (first >> 32) as u32 ^ counter[3] ^ key[1],
            first as u32,
        ];
        key[0] = key[0].wrapping_add(0x9e3779b9);
        key[1] = key[1].wrapping_add(0xbb67ae85);
    }
    let uniform = (counter[(block % 4) as usize] >> 8).max(1) as f32 / 16777216.0;
    -(-uniform.ln()).ln()
}

fn reference(
    logits: &[f32],
    seed: u64,
    mask: Option<&[u32]>,
    policy: Policy,
) -> u32 {
    let mut ranked = logits
        .iter()
        .enumerate()
        .filter(|(index, _)| mask.is_none_or(|bits| bits[index / 32] & (1 << (index % 32)) != 0))
        .map(|(index, value)| (index as u32, *value / policy.temperature.unwrap_or(1.0)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    let maximum = ranked.first().map_or(0.0, |value| value.1);
    let normalizer = ranked.iter().map(|value| (value.1 - maximum).exp()).sum::<f32>();
    let mut mass = 0.0;
    let mut winner = (0, f32::NEG_INFINITY);
    for (rank, (index, value)) in ranked.into_iter().enumerate() {
        if policy.top_k.is_some_and(|k| rank >= k as usize)
            || policy.top_p.is_some_and(|p| mass >= p)
            || policy.min_p.is_some_and(|p| value < maximum + p.ln())
        {
            break;
        }
        mass += (value - maximum).exp() / normalizer;
        let score = value
            + if policy.stochastic {
                gumbel(seed, index, logits.len() as u32)
            } else {
                0.0
            };
        if score > winner.1 || (score == winner.1 && index < winner.0) {
            winner = (index, score);
        }
    }
    winner.0
}

// Match the public CPU kernel's ascending iterator fold, independently of the
// GPU's ordered-bit representation and parallel reduction schedule.
fn greedy_reference(
    logits: &[f32],
    mask: Option<&[u32]>,
    dtype: DataType,
) -> u32 {
    logits
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let value = if mask.is_some_and(|bits| bits[index / 32] & (1 << (index % 32)) == 0) {
                f32::NEG_INFINITY
            } else if dtype == DataType::BF16 {
                f32::from_bits(value.to_bits() & 0xffff0000)
            } else {
                value
            };
            (index, value)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32
}

fn check_greedy(
    context: &GpuContext,
    name: &str,
    logits: &[f32],
    mask: Option<&[u32]>,
    vocab: usize,
    dtype: DataType,
) -> Result<(), String> {
    let batch = logits.len() / vocab;
    let mask_stride = vocab.div_ceil(32);
    let expected = (0..batch)
        .map(|b| {
            greedy_reference(
                &logits[b * vocab..(b + 1) * vocab],
                mask.map(|mask| &mask[b * mask_stride..(b + 1) * mask_stride]),
                dtype,
            )
        })
        .collect::<Vec<_>>();
    let actual = sample(context, logits, &vec![0; batch], mask, vocab as u32, dtype, Policy::default())?;
    if actual != expected {
        return Err(format!("{name} {dtype:?}: WebGPU {actual:?}, CPU argmax {expected:?}"));
    }
    Ok(())
}

fn greedy_cases(context: &GpuContext) -> Result<usize, String> {
    let mut cases = 0;
    for dtype in [DataType::F32, DataType::BF16] {
        // Both sides of the first partition boundary, the real Qwen scale, and
        // more than 256 partial maxima in the final workgroup. Ties span groups;
        // batch two's winner is the last element of an incomplete final group.
        for vocab in [1025, 248003, 262149] {
            let mut logits = vec![-4.0; vocab * 3];
            logits[1023] = 8.0;
            logits[1024] = 8.0;
            logits[vocab - 1] = 8.0;
            logits[vocab + vocab - 1] = 10.0;
            logits[vocab * 2 + 17] = 9.0;
            logits[vocab * 2 + 1024] = 9.0;
            check_greedy(context, "greedy partition/tie/tail", &logits, None, vocab, dtype)?;
            cases += 1;
        }

        let vocab = 8197;
        let mut logits = vec![-1.0; vocab * 4];
        logits[0] = f32::NAN;
        logits[1023] = 7.0;
        logits[1024] = 9.0;
        logits[vocab - 1] = 9.0;
        logits[vocab] = f32::INFINITY;
        logits[vocab * 2 + vocab - 1] = f32::NEG_INFINITY;
        logits[vocab * 3] = f32::NAN;
        let stride = vocab.div_ceil(32);
        let mut mask = vec![0u32; stride * 4];
        for index in [1023, 1024, vocab - 1] {
            mask[index / 32] |= 1 << (index % 32);
        }
        // Batch one is entirely masked; batch two permits only -infinity at the
        // last token (which ties masked index zero). Batch three keeps NaN zero.
        mask[stride * 2 + (vocab - 1) / 32] = 1 << ((vocab - 1) % 32);
        mask[stride * 3] = 1;
        check_greedy(context, "greedy batched masks", &logits, Some(&mask), vocab, dtype)?;
        cases += 1;

        for vocab in [769, 8197] {
            let mut logits = vec![f32::NEG_INFINITY; vocab * 6];
            logits[0] = f32::NAN;
            logits[vocab - 1] = f32::INFINITY;
            logits[vocab + 1] = f32::NAN;
            logits[vocab * 2 - 1] = 4.0;
            // Batch two stays -infinity throughout.
            logits[vocab * 3] = -0.0;
            logits[vocab * 4 - 1] = 0.0;
            logits[vocab * 4 + 2] = f32::INFINITY;
            logits[vocab * 5 - 1] = f32::INFINITY;
            // Keep subnormal ordering and BF16 truncation independent of flush
            // behavior in arithmetic: this operation compares source logits.
            logits[vocab * 5] = -f32::from_bits(1);
            logits[vocab * 5 + 1] = f32::from_bits(1);
            logits[vocab * 6 - 1] = f32::from_bits(0x00010000);
            check_greedy(context, "greedy NaN/infinity/zero/subnormal", &logits, None, vocab, dtype)?;
            cases += 1;
        }
    }
    Ok(cases)
}

fn check_stochastic(
    context: &GpuContext,
    name: &str,
    logits: &[f32],
    seeds: &[u64],
    mask: Option<&[u32]>,
    vocab: usize,
    dtype: DataType,
    policy: Policy,
) -> Result<(), String> {
    let quantized = logits
        .iter()
        .map(|&value| {
            if dtype == DataType::BF16 {
                f32::from_bits(value.to_bits() & 0xffff0000)
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let stride = vocab.div_ceil(32);
    let expected = seeds
        .iter()
        .enumerate()
        .map(|(batch, &seed)| {
            reference(
                &quantized[batch * vocab..(batch + 1) * vocab],
                seed,
                mask.map(|mask| &mask[batch * stride..(batch + 1) * stride]),
                policy,
            )
        })
        .collect::<Vec<_>>();
    let actual = sample(context, logits, seeds, mask, vocab as u32, dtype, policy)?;
    if actual != expected {
        return Err(format!("{name} {dtype:?}: WebGPU {actual:?}, CPU sorted reference {expected:?}"));
    }
    Ok(())
}

fn small_top_k_cases(context: &GpuContext) -> Result<usize, String> {
    let seeds = [0x0123_4567_89ab_cdef, 0xffff_ffff_9876_5432, 31, 279];
    let mut cases = 0;
    for dtype in [DataType::F32, DataType::BF16] {
        // Actual checkpoint defaults, with enough tiles to require two merge
        // levels. Tied logits span tiles and masked IDs exercise original-index
        // Philox mapping after selection. Normalization includes non-top-k IDs.
        for (name, vocab, k, temperature, top_p) in
            [("Qwen defaults", 248003, 20, 0.6, 0.95), ("LFM defaults", 65539, 50, 0.1, 0.1)]
        {
            let logits = (0..vocab * seeds.len())
                .map(|index| ((index * 43 + index / vocab * 17) % 509) as f32 / 128.0 - 1.0)
                .collect::<Vec<_>>();
            let stride = vocab.div_ceil(32);
            let mut mask = vec![0u32; stride * seeds.len()];
            for batch in 0..seeds.len() {
                for index in 0..vocab {
                    if (index + batch * 3) % 11 != 0 {
                        mask[batch * stride + index / 32] |= 1 << (index % 32);
                    }
                }
            }
            check_stochastic(
                context,
                name,
                &logits,
                &seeds,
                Some(&mask),
                vocab,
                dtype,
                Policy {
                    stochastic: true,
                    temperature: Some(temperature),
                    top_k: Some(k),
                    top_p: Some(top_p),
                    ..Policy::default()
                },
            )?;
            cases += 1;
        }
        for (vocab, k) in [(31, 64), (1025, 1), (33001, 64)] {
            let logits = (0..vocab * seeds.len())
                .map(|index| ((index * 17 + index / vocab) % 71) as f32 / 16.0 - 2.0)
                .collect::<Vec<_>>();
            check_stochastic(
                context,
                "small top-k tile/merge/tail",
                &logits,
                &seeds,
                None,
                vocab,
                dtype,
                Policy {
                    stochastic: true,
                    temperature: Some(0.75),
                    top_k: Some(k),
                    min_p: Some(0.4),
                    ..Policy::default()
                },
            )?;
            cases += 1;
        }

        // With 2048 equal logits the nucleus boundary and its neighboring
        // floats are exact. Choose a seed whose best of33 is ID32, so including
        // the crossing token incorrectly changes an observed output.
        let vocab = 2048;
        let boundary = 32.0f32 / vocab as f32;
        let crossing_seed = (0u64..4096)
            .find(|&seed| (0..32).all(|index| gumbel(seed, 32, vocab as u32) > gumbel(seed, index, vocab as u32)))
            .ok_or("no cutoff fixture seed")?;
        let cutoff_seeds = [crossing_seed, 1, 2, 3];
        let logits = vec![0.0; vocab * cutoff_seeds.len()];
        for p in [0.0, f32::from_bits(boundary.to_bits() - 1), boundary, f32::from_bits(boundary.to_bits() + 1), 1.0] {
            check_stochastic(
                context,
                "small top-k exact nucleus cutoff",
                &logits,
                &cutoff_seeds,
                None,
                vocab,
                dtype,
                Policy {
                    stochastic: true,
                    top_k: Some(64),
                    top_p: Some(p),
                    ..Policy::default()
                },
            )?;
            cases += 1;
        }

        let vocab = 4099;
        let mut logits = vec![-0.25; vocab * seeds.len()];
        for batch in 0..seeds.len() {
            for index in [0, 1023, 1024, 4098] {
                logits[batch * vocab + index] = 0.0;
            }
        }
        check_stochastic(
            context,
            "small top-k exact min-p boundary",
            &logits,
            &seeds,
            None,
            vocab,
            dtype,
            Policy {
                stochastic: true,
                top_k: Some(20),
                min_p: Some(1.0),
                ..Policy::default()
            },
        )?;
        cases += 1;
        let stride = vocab.div_ceil(32);
        let mut mask = vec![0u32; stride * seeds.len()];
        // Empty row, one permitted tail, and two sparse rows.
        mask[stride + (vocab - 1) / 32] = 1 << ((vocab - 1) % 32);
        for batch in 2..seeds.len() {
            for index in [0, 1023, 1024, 4098] {
                mask[batch * stride + index / 32] |= 1 << (index % 32);
            }
        }
        check_stochastic(
            context,
            "small top-k empty and sparse masks",
            &logits,
            &seeds,
            Some(&mask),
            vocab,
            dtype,
            Policy {
                stochastic: true,
                top_k: Some(50),
                top_p: Some(0.75),
                ..Policy::default()
            },
        )?;
        cases += 1;
    }
    // Finite source logits can overflow during temperature scaling. This row
    // exercises the entirely GPU-side existing-sampler fallback.
    check_stochastic(
        context,
        "small top-k nonfinite transformed fallback",
        &[1.0, f32::MAX, -1.0, f32::MAX, 1.0, f32::MAX],
        &[0, 1],
        None,
        3,
        DataType::F32,
        Policy {
            temperature: Some(f32::MIN_POSITIVE),
            top_k: Some(1),
            ..Policy::default()
        },
    )?;
    Ok(cases + 1)
}

pub(crate) fn run(context: &GpuContext) -> Result<Value, String> {
    let mut cases = greedy_cases(context)? + small_top_k_cases(context)?;
    for dtype in [DataType::F32, DataType::BF16] {
        let result =
            sample(context, &[1.0, 3.0, 3.0, -2.0, -1.0, 0.0, 5.0, 4.0], &[0, 1], None, 4, dtype, Policy::default())?;
        if result != [1, 2] {
            return Err(format!("{dtype:?} greedy/tie result {result:?}, expected [1, 2]"));
        }
        cases += 1;
    }

    let vocab = 1537;
    let batch = 3;
    let seeds = [0x0123_4567_89ab_cdef, 0xffff_ffff_9876_5432, 31];
    let logits = (0..vocab * batch)
        .map(|index| ((index * 37 + index / vocab * 11) % 97) as f32 * 0.0625 - 3.0)
        .collect::<Vec<_>>();
    let mask_stride = (vocab as u32).div_ceil(32) as usize;
    let mut mask = vec![0u32; mask_stride * batch];
    for b in 0..batch {
        for index in 0..vocab {
            if (index + b) % 7 != 0 {
                mask[b * mask_stride + index / 32] |= 1 << (index % 32);
            }
        }
    }
    let stochastic = Policy {
        stochastic: true,
        temperature: Some(0.75),
        ..Policy::default()
    };
    for (name, policy) in [
        ("stochastic", stochastic),
        (
            "top-k-one",
            Policy {
                top_k: Some(1),
                ..stochastic
            },
        ),
        (
            "top-k",
            Policy {
                top_k: Some(19),
                ..stochastic
            },
        ),
        (
            "top-p",
            Policy {
                top_p: Some(0.63),
                ..stochastic
            },
        ),
        (
            "min-p",
            Policy {
                min_p: Some(0.41),
                ..stochastic
            },
        ),
        (
            "combined",
            Policy {
                top_k: Some(83),
                top_p: Some(0.78),
                min_p: Some(0.18),
                ..stochastic
            },
        ),
    ] {
        let expected = (0..batch)
            .map(|b| {
                reference(
                    &logits[b * vocab..(b + 1) * vocab],
                    seeds[b],
                    Some(&mask[b * mask_stride..(b + 1) * mask_stride]),
                    policy,
                )
            })
            .collect::<Vec<_>>();
        let actual = sample(context, &logits, &seeds, Some(&mask), vocab as u32, DataType::F32, policy)?;
        if actual != expected {
            return Err(format!("{name}: WebGPU {actual:?}, CPU reference {expected:?}"));
        }
        cases += 1;
    }

    // A repeated token is penalized once, regardless of how often it appears.
    let original = [2.0f32, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0];
    let mut source = context.create_buffer(32).map_err(|e| e.to_string())?;
    source.copyin(&original);
    let mut penalized = context.create_buffer(32).map_err(|e| e.to_string())?;
    let mut ring = context.create_buffer(24).map_err(|e| e.to_string())?;
    ring.copyin(&[1u32, 3, 0, 1, 2, 1]);
    let mut input = context.create_buffer(12).map_err(|e| e.to_string())?;
    input.copyin(&[3u32, 3, 0]);
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    command.encode_copy(&source, &mut penalized);
    PenaltyKernel::new(context, DataType::F32).map_err(|e| e.to_string())?.encode(
        &source,
        &mut penalized,
        &ring,
        &input,
        2.0,
        4,
        4,
        1,
        2,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let expected = [2.0, -4.0, 2.0, -8.0, 3.0, -12.0, 8.0, -16.0];
    let actual = penalized.copyout::<f32>();
    if actual != expected {
        return Err(format!("Repetition penalty: {actual:?}, expected {expected:?}"));
    }
    cases += 1;

    let mut additions = context.create_buffer(24).map_err(|e| e.to_string())?;
    additions.copyin(&[5u32, 6, 7, 8, 9, 10]);
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    RingKernel::new(context).map_err(|e| e.to_string())?.encode(&additions, &mut ring, 4, 6, &mut command);
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let actual = ring.copyout::<u32>();
    let expected = [2, 4, 9, 10, 7, 8];
    if actual != expected {
        return Err(format!("Context ring: {actual:?}, expected {expected:?}"));
    }
    cases += 1;
    Ok(json!({ "cases": cases, "reference": "CPU Philox and sampling semantics",
        "roundingNote": "Parallel full-vocabulary softmax sums can differ by floating-point rounding at non-exact nucleus boundaries; exact binary cutoffs are covered." }))
}
