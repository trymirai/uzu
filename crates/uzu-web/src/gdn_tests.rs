//! Independent scalar fixtures for the WebGPU recurrent state and prefill paths.
use serde_json::{Value, json};
use uzu_engine::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            kernel::{
                Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
                DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel, ShortConvPackKernel,
                ShortConvPrefillKernel,
            },
        },
        webgpu::WebGpu,
    },
    data_type::DataType,
};

type GpuContext = <WebGpu as Backend>::Context;
type Buffer = <WebGpu as Backend>::GlobalBuffer;
type Update = <<WebGpu as Backend>::Kernels as Kernels>::DeltaNetUpdateKernel;
type Prep = <<WebGpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel;
type Prefill = <<WebGpu as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel;
type NormGate = <<WebGpu as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel;
type ConvUpdate = <<WebGpu as Backend>::Kernels as Kernels>::DeltaNetConvUpdateKernel;
type ConvPack = <<WebGpu as Backend>::Kernels as Kernels>::Conv1dPackKernel;
type ConvScan = <<WebGpu as Backend>::Kernels as Kernels>::DeltaNetConvScanKernel;
type ShortPack = <<WebGpu as Backend>::Kernels as Kernels>::ShortConvPackKernel;
type ShortPrefill = <<WebGpu as Backend>::Kernels as Kernels>::ShortConvPrefillKernel;

const DK: usize = 128;
const DV: usize = 8;
const HV: usize = 2;
const HK: usize = 1;
const KEY: usize = DK * HK;
const VALUE: usize = DV * HV;
const CONV: usize = 2 * KEY + VALUE;
const PROJ: usize = CONV + VALUE + 2 * HV;
const EPSILON: f32 = 1e-6;
const A_LOG: [f32; HV] = [-0.5, 0.375];
const DT_BIAS: [f32; HV] = [0.125, -0.25];

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn cast(
    value: f32,
    dtype: DataType,
) -> f32 {
    if dtype == DataType::BF16 {
        f32::from_bits(u32::from(bf16_bits(value)) << 16)
    } else {
        value
    }
}

fn buffer(
    context: &GpuContext,
    values: &[f32],
    dtype: DataType,
) -> Result<Buffer, String> {
    let mut result = context.create_buffer(values.len() * dtype.size_in_bytes()).map_err(|e| e.to_string())?;
    if dtype == DataType::BF16 {
        result.copyin(&values.iter().copied().map(bf16_bits).collect::<Vec<_>>());
    } else {
        result.copyin(values);
    }
    Ok(result)
}

fn read(
    buffer: &Buffer,
    dtype: DataType,
) -> Vec<f32> {
    if dtype == DataType::BF16 {
        buffer.copyout::<u16>().into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect()
    } else {
        buffer.copyout::<f32>()
    }
}

fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length {} != {}", actual.len(), expected.len()));
    }
    let mut maximum = 0.0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let error = (a - b).abs();
        maximum = maximum.max(error);
        if !a.is_finite() || error > tolerance + tolerance * b.abs() {
            return Err(format!("{label}[{i}]: WebGPU {a}, reference {b}, error {error}"));
        }
    }
    Ok(maximum)
}

fn projections(
    tokens: usize,
    dtype: DataType,
) -> Vec<f32> {
    (0..tokens * PROJ).map(|i| cast(((i * 17 + i / PROJ * 7) % 61) as f32 / 32.0 - 0.75, dtype)).collect()
}

fn state() -> Vec<f32> {
    (0..VALUE * DK).map(|i| ((i * 13) % 43) as f32 / 256.0 - 0.078125).collect()
}

fn weights() -> Vec<f32> {
    (0..DV).map(|i| 0.625 + i as f32 / 16.0).collect()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

struct Prepared {
    q: Vec<f32>,
    k: Vec<f32>,
    beta: Vec<f32>,
    log_decay: Vec<f32>,
}

fn prepare_reference(input: &[f32]) -> Prepared {
    let tokens = input.len() / PROJ;
    let mut result = Prepared {
        q: vec![0.0; tokens * KEY],
        k: vec![0.0; tokens * KEY],
        beta: vec![0.0; tokens * HV],
        log_decay: vec![0.0; tokens * HV],
    };
    for t in 0..tokens {
        for hk in 0..HK {
            let q = &input[t * PROJ + hk * DK..t * PROJ + (hk + 1) * DK];
            let k = &input[t * PROJ + KEY + hk * DK..t * PROJ + KEY + (hk + 1) * DK];
            let q_inv = (q.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip();
            let k_inv = (k.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip();
            for j in 0..DK {
                result.q[t * KEY + hk * DK + j] = q[j] * q_inv * (DK as f32).sqrt().recip();
                result.k[t * KEY + hk * DK + j] = k[j] * k_inv;
            }
        }
        for hv in 0..HV {
            result.beta[t * HV + hv] = (1.0 + (-input[t * PROJ + CONV + VALUE + hv]).exp()).recip();
            let a = input[t * PROJ + CONV + VALUE + HV + hv] + DT_BIAS[hv];
            let softplus = if a > 20.0 {
                a
            } else {
                (1.0 + a.exp()).ln()
            };
            result.log_decay[t * HV + hv] = -A_LOG[hv].exp() * softplus;
        }
    }
    result
}

fn recurrent_reference(
    input: &[f32],
    dtype: DataType,
    prefill: bool,
) -> (Vec<f32>, Vec<f32>) {
    let prepared = prepare_reference(input);
    let mut state = state();
    let mut out = vec![0.0; input.len() / PROJ * VALUE];
    let norm_weight = weights();
    for t in 0..input.len() / PROJ {
        for hv in 0..HV {
            let hk = hv / (HV / HK);
            let q = &prepared.q[t * KEY + hk * DK..t * KEY + (hk + 1) * DK];
            let k = &prepared.k[t * KEY + hk * DK..t * KEY + (hk + 1) * DK];
            let decay = prepared.log_decay[t * HV + hv].exp();
            let beta = prepared.beta[t * HV + hv];
            let kq = k.iter().zip(q).map(|(k, q)| k * q).sum::<f32>();
            let mut raw = [0.0; DV];
            for i in 0..DV {
                let row = &mut state[(hv * DV + i) * DK..(hv * DV + i + 1) * DK];
                let value = input[t * PROJ + 2 * KEY + hv * DV + i];
                let retrieved = if prefill {
                    row.iter().zip(k).map(|(s, k)| (decay * s) * k).sum::<f32>()
                } else {
                    decay * row.iter().zip(k).map(|(s, k)| s * k).sum::<f32>()
                };
                let delta = beta * (value - retrieved);
                let mut output = decay * row.iter().zip(q).map(|(s, q)| s * q).sum::<f32>() + delta * kq;
                for j in 0..DK {
                    row[j] = decay * row[j] + k[j] * delta;
                }
                if prefill {
                    // Prefill stores to the activation dtype before the separate RMSNorm kernel.
                    output = cast(row.iter().zip(q).map(|(s, q)| s * q).sum::<f32>(), dtype);
                }
                raw[i] = output;
            }
            let inv_rms = (raw.iter().map(|v| v * v).sum::<f32>() / DV as f32 + EPSILON).sqrt().recip();
            for i in 0..DV {
                out[t * VALUE + hv * DV + i] =
                    cast(raw[i] * inv_rms * norm_weight[i] * silu(input[t * PROJ + CONV + hv * DV + i]), dtype);
            }
        }
    }
    (out, state)
}

fn recurrent(
    context: &GpuContext,
    dtype: DataType,
    prefill: bool,
) -> Result<Value, String> {
    let tokens = if prefill {
        5
    } else {
        1
    };
    let input = projections(tokens, dtype);
    let input_buffer = buffer(context, &input, dtype)?;
    let a = buffer(context, &A_LOG, DataType::F32)?;
    let dt = buffer(context, &DT_BIAS, DataType::F32)?;
    let weight = buffer(context, &weights(), DataType::F32)?;
    let mut state_buffer = buffer(context, &state(), DataType::F32)?;
    let mut out = buffer(context, &vec![0.0; tokens * VALUE], dtype)?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    if prefill {
        let mut q = buffer(context, &vec![0.0; tokens * KEY], DataType::F32)?;
        let mut k = buffer(context, &vec![0.0; tokens * KEY], DataType::F32)?;
        let mut beta = buffer(context, &vec![0.0; tokens * HV], DataType::F32)?;
        let mut decay = buffer(context, &vec![0.0; tokens * HV], DataType::F32)?;
        Prep::new(context, dtype, DataType::F32, DK as u32, false, false).map_err(|e| e.to_string())?.encode(
            &input_buffer,
            &a,
            &dt,
            &mut q,
            &mut k,
            None::<&mut Buffer>,
            &mut beta,
            &mut decay,
            HV as u32,
            HK as u32,
            KEY as u32,
            VALUE as u32,
            tokens as u32,
            &mut command,
        );
        Prefill::new(context, dtype, DK as u32).map_err(|e| e.to_string())?.encode(
            &q,
            &k,
            &beta,
            &decay,
            &input_buffer,
            &mut state_buffer,
            &mut out,
            HV as u32,
            HK as u32,
            DV as u32,
            KEY as u32,
            VALUE as u32,
            tokens as u32,
            1,
            &mut command,
        );
        NormGate::new(context, dtype).map_err(|e| e.to_string())?.encode(
            &mut out,
            &input_buffer,
            &weight,
            HV as u32,
            DV as u32,
            VALUE as u32,
            CONV as u32,
            PROJ as u32,
            EPSILON,
            tokens as u32,
            &mut command,
        );
        command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        let expected = prepare_reference(&input);
        compare("GDN prep Q", &read(&q, DataType::F32), &expected.q, 2e-6)?;
        compare("GDN prep K", &read(&k, DataType::F32), &expected.k, 2e-6)?;
        compare("GDN prep beta", &read(&beta, DataType::F32), &expected.beta, 2e-6)?;
        compare(
            "GDN prep decay",
            &read(&decay, DataType::F32),
            &expected.log_decay.iter().map(|v| v.exp()).collect::<Vec<_>>(),
            2e-6,
        )?;
    } else {
        Update::new(context, dtype, DK as u32).map_err(|e| e.to_string())?.encode(
            &input_buffer,
            &a,
            &dt,
            &weight,
            &mut state_buffer,
            &mut out,
            HV as u32,
            HK as u32,
            DV as u32,
            KEY as u32,
            VALUE as u32,
            EPSILON,
            &mut command,
        );
        command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    }
    let (expected_out, expected_state) = recurrent_reference(&input, dtype, prefill);
    let label = format!(
        "GDN {} {dtype:?}",
        if prefill {
            "prefill"
        } else {
            "update"
        }
    );
    let output_error = compare(
        &label,
        &read(&out, dtype),
        &expected_out,
        if dtype == DataType::BF16 {
            0.004
        } else {
            3e-5
        },
    )?;
    let state_error = compare(&format!("{label} state"), &read(&state_buffer, DataType::F32), &expected_state, 4e-6)?;
    Ok(json!({"name":label,"tokens":tokens,"outputMaxError":output_error,"stateMaxError":state_error}))
}

fn prep_variants(context: &GpuContext) -> Result<Value, String> {
    let tokens = 3;
    let mut input = projections(tokens, DataType::BF16);
    // Cover the stable softplus branch used by extreme gate inputs.
    input[CONV + VALUE + HV] = 24.0;
    let source = buffer(context, &input, DataType::BF16)?;
    let a = buffer(context, &A_LOG, DataType::F32)?;
    let dt = buffer(context, &DT_BIAS, DataType::F32)?;
    let mut q = buffer(context, &vec![0.0; tokens * KEY], DataType::BF16)?;
    let mut k = buffer(context, &vec![0.0; tokens * KEY], DataType::BF16)?;
    let mut v = buffer(context, &vec![0.0; tokens * VALUE], DataType::BF16)?;
    let mut beta = buffer(context, &vec![0.0; tokens * HV], DataType::F32)?;
    let mut decay = buffer(context, &vec![0.0; tokens * HV], DataType::F32)?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    Prep::new(context, DataType::BF16, DataType::BF16, DK as u32, true, true).map_err(|e| e.to_string())?.encode(
        &source,
        &a,
        &dt,
        &mut q,
        &mut k,
        Some(&mut v),
        &mut beta,
        &mut decay,
        HV as u32,
        HK as u32,
        KEY as u32,
        VALUE as u32,
        tokens as u32,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let expected = prepare_reference(&input);
    for (label, actual, expected) in [("Q", &q, &expected.q), ("K", &k, &expected.k)] {
        compare(
            label,
            &read(actual, DataType::BF16),
            &expected.iter().map(|v| cast(*v, DataType::BF16)).collect::<Vec<_>>(),
            0.001,
        )?;
    }
    compare("GDN log decay", &read(&decay, DataType::F32), &expected.log_decay, 2e-6)?;
    compare(
        "GDN compact V",
        &read(&v, DataType::BF16),
        &(0..tokens)
            .flat_map(|t| input[t * PROJ + 2 * KEY..t * PROJ + 2 * KEY + VALUE].iter().copied())
            .collect::<Vec<_>>(),
        0.0,
    )?;
    Ok(json!({"name":"GDN BF16 prep with log decay and compact V"}))
}

fn convolution(
    context: &GpuContext,
    dtype: DataType,
    scan: bool,
    has_bias: bool,
) -> Result<Value, String> {
    let channels = 10;
    let stride = channels + 6;
    let taps = 4;
    let history = taps - 1;
    let tokens = if scan {
        5
    } else {
        1
    };
    let input = (0..tokens * stride).map(|i| cast(((i * 11) % 31) as f32 / 32.0 - 0.375, dtype)).collect::<Vec<_>>();
    let weights = (0..channels * taps).map(|i| ((i * 5) % 13) as f32 / 32.0 - 0.125).collect::<Vec<_>>();
    let bias = (0..channels).map(|i| i as f32 / 128.0 - 0.03125).collect::<Vec<_>>();
    let initial = (0..channels * history).map(|i| (i % 17) as f32 / 64.0 - 0.09375).collect::<Vec<_>>();
    let mut expected = input.clone();
    let mut expected_state = initial.clone();
    for t in 0..tokens {
        for c in 0..channels {
            let x = input[t * stride + c];
            let row = &mut expected_state[c * history..(c + 1) * history];
            let mut sum = if has_bias {
                bias[c]
            } else {
                0.0
            };
            for tap in 0..history {
                sum += row[tap] * weights[c * taps + tap];
            }
            sum += x * weights[c * taps + history];
            expected[t * stride + c] = cast(silu(sum), dtype);
            row.rotate_left(1);
            row[history - 1] = x;
        }
    }
    let weight = buffer(context, &weights, DataType::F32)?;
    let bias_buffer = buffer(context, &bias, DataType::F32)?;
    let mut in_out = buffer(context, &input, dtype)?;
    let mut state_buffer = buffer(context, &initial, DataType::F32)?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    if scan {
        let mut padded = buffer(context, &vec![0.0; (tokens + history) * stride], DataType::F32)?;
        ConvPack::new(context, DataType::F32, dtype).map_err(|e| e.to_string())?.encode(
            &state_buffer,
            &in_out,
            &mut padded,
            history as u32,
            stride as u32,
            tokens as u32,
            channels as u32,
            &mut command,
        );
        ConvScan::new(context, dtype, has_bias).map_err(|e| e.to_string())?.encode(
            &padded,
            &weight,
            has_bias.then_some(&bias_buffer),
            &mut in_out,
            &mut state_buffer,
            tokens as u32,
            taps as u32,
            stride as u32,
            history as u32,
            channels as u32,
            stride as u32,
            &mut command,
        );
    } else {
        ConvUpdate::new(context, dtype, has_bias).map_err(|e| e.to_string())?.encode(
            &weight,
            has_bias.then_some(&bias_buffer),
            &mut in_out,
            &mut state_buffer,
            taps as u32,
            channels as u32,
            history as u32,
            &mut command,
        );
    }
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let label = format!(
        "GDN conv {} {dtype:?} bias={has_bias}",
        if scan {
            "pack+scan"
        } else {
            "update"
        }
    );
    let error = compare(
        &label,
        &read(&in_out, dtype),
        &expected,
        if dtype == DataType::BF16 {
            0.001
        } else {
            2e-6
        },
    )?;
    compare(&format!("{label} state"), &read(&state_buffer, DataType::F32), &expected_state, 0.0)?;
    Ok(json!({"name":label,"outputMaxError":error}))
}

pub(crate) fn run(context: &GpuContext) -> Result<Value, String> {
    let mut cases = Vec::new();
    for dtype in [DataType::F32, DataType::BF16] {
        cases.push(recurrent(context, dtype, false)?);
        cases.push(recurrent(context, dtype, true)?);
        for scan in [false, true] {
            for bias in [false, true] {
                cases.push(convolution(context, dtype, scan, bias)?);
            }
        }
        for has_bias in [false, true] {
            cases.push(short_convolution(context, dtype, 6, 129, 4, has_bias)?);
        }
    }
    cases.push(short_convolution(context, DataType::F32, 1, 5, 1, false)?);
    cases.push(short_convolution(context, DataType::BF16, 4 * 65535 + 5, 1, 2, false)?);
    cases.push(prep_variants(context)?);
    Ok(json!({"cases":cases,"reference":"scalar recurrent state, normalization and causal convolution"}))
}

fn short_convolution(
    context: &GpuContext,
    dtype: DataType,
    tokens: usize,
    channels: usize,
    taps: usize,
    has_bias: bool,
) -> Result<Value, String> {
    let history = taps - 1;
    let stride = 3 * channels + 2;
    let input: Vec<_> = (0..tokens * stride).map(|i| cast((i * 11 % 31) as f32 / 32.0 - 0.375, dtype)).collect();
    let weights: Vec<_> = (0..channels * taps).map(|i| (i * 5 % 13) as f32 / 32.0 - 0.125).collect();
    let bias: Vec<_> = (0..channels).map(|i| (i % 7) as f32 / 128.0 - 0.03125).collect();
    // A one-tap convolution has no history; keep a sentinel allocation so the
    // unused binding still satisfies the backend's nonempty-buffer contract.
    let initial: Vec<_> =
        (0..(channels * history).max(1)).map(|i| cast((i % 17) as f32 / 64.0 - 0.09375, dtype)).collect();
    let mut expected_padded = vec![0.0; (tokens + history) * channels];
    for row in 0..tokens + history {
        for channel in 0..channels {
            expected_padded[row * channels + channel] = if row < history {
                initial[channel * history + row]
            } else {
                let offset = (row - history) * stride + channel;
                cast(input[offset] * input[offset + 2 * channels], dtype)
            };
        }
    }
    // A nonzero output offset and odd channel counts exercise packed words
    // next to guard values while neighboring token tiles run independently.
    let mut expected = vec![-256.0; 1 + tokens * channels + 1];
    for token in 0..tokens {
        for channel in 0..channels {
            let mut value = if has_bias {
                bias[channel]
            } else {
                0.0
            };
            for tap in 0..taps {
                value += weights[channel * taps + tap] * expected_padded[(token + tap) * channels + channel];
            }
            expected[1 + token * channels + channel] = cast(value * input[token * stride + channels + channel], dtype);
        }
    }
    let mut expected_state = initial.clone();
    for channel in 0..channels {
        for tap in 0..history {
            expected_state[channel * history + tap] = expected_padded[(tokens + tap) * channels + channel];
        }
    }
    let input_buffer = buffer(context, &input, dtype)?;
    let weights_buffer = buffer(context, &weights, DataType::F32)?;
    let bias_buffer = buffer(context, &bias, DataType::F32)?;
    let mut state = buffer(context, &initial, dtype)?;
    let mut padded = buffer(context, &vec![0.0; expected_padded.len()], dtype)?;
    let mut output = buffer(context, &vec![-256.0; expected.len()], dtype)?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    ShortPack::new(context, dtype).map_err(|e| e.to_string())?.encode(
        &state,
        &input_buffer,
        &mut padded,
        history as u32,
        tokens as u32,
        stride as u32,
        channels as u32,
        &mut command,
    );
    let bytes = dtype.size_in_bytes();
    ShortPrefill::new(context, dtype, DataType::F32, has_bias).map_err(|e| e.to_string())?.encode(
        &padded,
        &input_buffer,
        &weights_buffer,
        has_bias.then_some(&bias_buffer),
        (&mut output).subrange_mut(bytes..bytes * (1 + tokens * channels)),
        &mut state,
        tokens as u32,
        taps as u32,
        stride as u32,
        history as u32,
        channels as u32,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let name =
        format!("ShortConv pack+prefill {dtype:?} tokens={tokens} channels={channels} taps={taps} bias={has_bias}");
    compare(&format!("{name} padded"), &read(&padded, dtype), &expected_padded, 0.0)?;
    compare(&format!("{name} state"), &read(&state, dtype), &expected_state, 0.0)?;
    compare(&name, &read(&output, dtype), &expected, 0.0)?;
    Ok(json!({"name":name,"values":expected.len()}))
}
