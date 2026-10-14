//! Browser numerical fixtures, compiled only by the `kernel-tests` feature.
use serde_json::{Value, json};
use uzu_engine::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::{
                ActivationTransformOp, ActivationType, EmbeddingTableKind, QuantizationMethod, QuantizationMode,
            },
            kernel::{
                ActivationTransformKernel, AttentionSinglePassKernel, GatedActMul, GatedActMulSettings,
                InputEmbeddingLookupKernel, NormalizationKernel, ShortConvDecodeKernel, SoftmaxKernel,
                TensorAddScaleKernel,
                matmul::{
                    MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput, QuantParams,
                    QuantParamsLayout, QuantizedB, QuantizedCorrection,
                },
            },
        },
        webgpu::WebGpu,
    },
    data_type::DataType,
};
use wasm_bindgen::prelude::*;

type GpuContext = <WebGpu as Backend>::Context;
type Buffer = <WebGpu as Backend>::GlobalBuffer;
type Matmul = <<WebGpu as Backend>::Kernels as Kernels>::MatmulKernel;
type Attention = <<WebGpu as Backend>::Kernels as Kernels>::AttentionSinglePassKernel;
type AddScale = <<WebGpu as Backend>::Kernels as Kernels>::TensorAddScaleKernel;
type ShortConv = <<WebGpu as Backend>::Kernels as Kernels>::ShortConvDecodeKernel;
type Normalization = <<WebGpu as Backend>::Kernels as Kernels>::NormalizationKernel;
type Embedding = <<WebGpu as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel;
type Transform = <<WebGpu as Backend>::Kernels as Kernels>::ActivationTransformKernel;
type Softmax = <<WebGpu as Backend>::Kernels as Kernels>::SoftmaxKernel;

#[wasm_bindgen(js_name = webGpuProfile)]
pub fn webgpu_profile(reset: bool) -> String {
    uzu_engine::backends::webgpu::profile_snapshot(reset)
}

#[wasm_bindgen(js_name = testWebGpuKernels)]
pub async fn test_webgpu_kernels() -> Result<String, JsValue> {
    uzu_engine::backends::webgpu::prepare().await.map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = (|| -> Result<Value, String> {
        let context = GpuContext::new().map_err(|e| e.to_string())?;
        let mut cases = Vec::new();
        for dtype in [DataType::F32, DataType::F16, DataType::BF16] {
            cases.push(aliased_add(&context, dtype)?);
        }
        cases.push(quant_matmul(&context, 1, QuantParamsLayout::OutputGroup)?);
        cases.push(quant_matmul(&context, 17, QuantParamsLayout::GroupOutput)?);
        cases.push(full_matmul(&context, false)?);
        cases.push(full_matmul(&context, true)?);
        for dtype in [DataType::F32, DataType::BF16] {
            for dim in [64, 256] {
                cases.push(attention(&context, dtype, dim, 3, 2)?);
                cases.push(attention(&context, dtype, dim, 1, 38)?);
                cases.push(masked_ring_attention(&context, dtype, dim)?);
            }
        }
        cases.push(short_conv(&context)?);
        cases.push(clipped_gate(&context)?);
        for dtype in [DataType::F32, DataType::BF16] {
            for has_sinks in [false, true] {
                cases.push(softmax(&context, dtype, has_sinks)?);
            }
        }
        for residual in [false, true] {
            for in_place in [false, true] {
                cases.push(normalization(&context, residual, in_place)?);
            }
        }
        for columns in [1, 255, 256, 257, 768, 1025, 2049, 8192, 8193] {
            cases.push(normalization_shape_edges(&context, columns, false, false, columns % 2 == 1)?);
        }
        for (copy, residual) in [(true, false), (true, true), (false, true)] {
            cases.push(normalization_shape_edges(&context, 513, copy, residual, true)?);
        }
        for dtype in [DataType::F32, DataType::BF16] {
            for quantized in [false, true] {
                cases.push(embedding(&context, dtype, quantized)?);
            }
            for in_place in [false, true] {
                for op in [ActivationTransformOp::InputRht, ActivationTransformOp::OutputRht] {
                    cases.push(activation_transform(&context, dtype, in_place, op)?);
                }
            }
        }
        Ok(
            json!({"device":context.device_name(),"kernels":cases,"sampling":crate::sampling_tests::run(&context)?,"gdn":crate::gdn_tests::run(&context)?,"gemv":crate::gemv_tests::run(&context)?,"shaderCache":crate::shader_cache_tests::run(&context)?}),
        )
    })();
    result.map(|v| v.to_string()).map_err(|e| JsValue::from_str(&e))
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
fn bf16(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(value)) << 16)
}
fn buffer_f32(
    context: &GpuContext,
    values: &[f32],
) -> Result<Buffer, String> {
    let mut buffer = context.create_buffer(values.len() * 4).map_err(|e| e.to_string())?;
    buffer.copyin(values);
    Ok(buffer)
}
fn buffer_u16(
    context: &GpuContext,
    values: &[u16],
) -> Result<Buffer, String> {
    let mut buffer = context.create_buffer(values.len() * 2).map_err(|e| e.to_string())?;
    buffer.copyin(values);
    Ok(buffer)
}
fn buffer_bf16(
    context: &GpuContext,
    values: &[f32],
) -> Result<Buffer, String> {
    buffer_u16(context, &values.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>())
}
fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<Value, String> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: length {} does not match expected {}", actual.len(), expected.len()));
    }
    let mut error = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).abs();
        error = error.max(difference);
        if !actual.is_finite() || difference > tolerance * (1.0 + expected.abs()) {
            return Err(format!("{name}: index {index} actual {actual} expected {expected}, difference {difference}"));
        }
    }
    Ok(json!({"name":name,"maxAbsoluteError":error,"values":actual.len()}))
}

fn clipped_gate(context: &GpuContext) -> Result<Value, String> {
    let values = [-4.0f32, -1.0, 2.0, 5.0];
    let gates = [-3.0f32, -0.5, 1.0, 4.0];
    let expected: Vec<f32> = values
        .into_iter()
        .zip(gates)
        .map(|(value, gate)| {
            let value = bf16(value.clamp(-2.21, 3.31));
            let gate = bf16(gate.clamp(-1.13, 2.37));
            bf16(value * bf16(gate / (1.0 + (-0.5 * gate).exp())))
        })
        .collect();
    let input = buffer_bf16(context, &values.into_iter().chain(gates).collect::<Vec<_>>())?;
    let mut output = buffer_bf16(context, &[0.0; 4])?;
    let kernel = GatedActMul::<WebGpu>::full_precision(
        context,
        DataType::BF16,
        true,
        false,
        GatedActMulSettings {
            activation_alpha: Some(0.5),
            gate_clipping: serde_json::from_str("[-1.13,2.37]").map_err(|e| e.to_string())?,
            value_clipping: serde_json::from_str("[-2.21,3.31]").map_err(|e| e.to_string())?,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode_fp(
        &input,
        None::<&Buffer>,
        &mut output,
        None::<&Buffer>,
        4,
        1,
        0,
        0,
        ActivationType::SILU,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let actual: Vec<_> = output.copyout::<u16>().iter().map(|&v| f32::from_bits(u32::from(v) << 16)).collect();
    compare("BF16 gated activation with fractional clipping and custom alpha", &actual, &expected, 0.0)
}

fn softmax(
    context: &GpuContext,
    dtype: DataType,
    has_sinks: bool,
) -> Result<Value, String> {
    let (rows, batch, width) = (2usize, 3usize, 513usize);
    let input: Vec<_> = (0..rows * batch * width)
        .map(|i| {
            let v = ((i * 37 % 83) as f32 - 41.0) / 9.0;
            if dtype == DataType::BF16 {
                bf16(v)
            } else {
                v
            }
        })
        .collect();
    let sinks = [-1.5f32, 3.25];
    let mut expected = Vec::with_capacity(input.len());
    for (row, values) in input.chunks_exact(width).enumerate() {
        let maximum = values.iter().copied().fold(
            if has_sinks {
                sinks[row / batch]
            } else {
                f32::NEG_INFINITY
            },
            f32::max,
        );
        let sum = values.iter().map(|value| (value - maximum).exp()).sum::<f32>()
            + if has_sinks {
                (sinks[row / batch] - maximum).exp()
            } else {
                0.0
            };
        expected.extend(values.iter().map(|value| {
            let probability = (value - maximum).exp();
            if dtype == DataType::BF16 {
                bf16(bf16(probability) / sum)
            } else {
                probability / sum
            }
        }));
    }
    let mut values = if dtype == DataType::BF16 {
        buffer_bf16(context, &input)?
    } else {
        buffer_f32(context, &input)?
    };
    let sinks = if dtype == DataType::BF16 {
        buffer_bf16(context, &sinks)?
    } else {
        buffer_f32(context, &sinks)?
    };
    let kernel = Softmax::new(context, dtype, has_sinks).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(&mut values, has_sinks.then_some(&sinks), width as u32, rows as u32, batch as u32, &mut command);
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let actual = if dtype == DataType::BF16 {
        values.copyout::<u16>().iter().map(|&v| f32::from_bits(u32::from(v) << 16)).collect()
    } else {
        values.copyout::<f32>()
    };
    compare(&format!("Softmax {dtype:?} sinks={has_sinks}"), &actual, &expected, 1e-6)
}

fn aliased_add(
    context: &GpuContext,
    dtype: DataType,
) -> Result<Value, String> {
    let mut data = match dtype {
        DataType::F32 => buffer_f32(context, &[1., 2., 3., 4., 5., 6., 7., 8.])?,
        DataType::F16 => buffer_u16(context, &[0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800])?,
        _ => buffer_bf16(context, &[1., 2., 3., 4., 5., 6., 7., 8.])?,
    };
    let kernel = AddScale::new(context, dtype, true).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    // Both arguments share one physical allocation; the second view starts at
    // byte 6 for half types, exercising deduplication and unaligned half offsets.
    let (bias, mut output) = (&mut data).split_at(3 * dtype.size_in_bytes());
    kernel.encode(None::<&Buffer>, bias.as_ref(), output.reborrow(), 3, 5, 0.5, &mut command);
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let actual = match dtype {
        DataType::F32 => data.copyout::<f32>(),
        DataType::BF16 => data.copyout::<u16>().iter().map(|&v| f32::from_bits(u32::from(v) << 16)).collect(),
        _ => data
            .copyout::<u16>()
            .iter()
            .map(|&v| {
                let exponent = (v >> 10) & 31;
                let fraction = v & 1023;
                (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15)
            })
            .collect(),
    };
    compare(&format!("aliased offset add {dtype:?}"), &actual, &[1., 2., 3., 2.5, 3.5, 4.5, 4., 5.], 0.0)
}

fn quant_matmul(
    context: &GpuContext,
    m: u32,
    layout: QuantParamsLayout,
) -> Result<Value, String> {
    let (n, k) = (19u32, 64u32);
    let a_values: Vec<f32> = (0..m * k + 3).map(|i| bf16(((i * 7 % 29) as f32 - 14.0) / 16.0)).collect();
    let a = buffer_bf16(context, &a_values)?;
    let metadata = QuantParams::new(layout, n, 2);
    let shape = metadata.scale_shape();
    let strides = metadata.scale_strides();
    let mut scale_values = vec![0.0; shape[0] as usize * shape[1] as usize];
    for col in 0..n {
        for group in 0..2 {
            scale_values[(col * strides.output_stride + group * strides.group_stride) as usize] =
                bf16((col % 5 + group + 1) as f32 / 32.0);
        }
    }
    let scales = buffer_bf16(context, &scale_values)?;
    let mut codes = vec![0u32; (n * k / 8) as usize];
    for index in 0..n * k {
        let code = (index * 13 + 7) % 16;
        codes[(index / 8) as usize] |= (code ^ 8) << ((index % 8) * 4);
    }
    let mut code_buffer = context.create_buffer(codes.len() * 4).map_err(|e| e.to_string())?;
    code_buffer.copyin(&codes);
    let stride = n + 3;
    let mut output = buffer_f32(context, &vec![-999.0; (m * stride) as usize])?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    Matmul::new(context, DataType::BF16, DataType::BF16, DataType::F32)
        .map_err(|e| e.to_string())?
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a,
                    offset: 3,
                },
                b: MatmulB::Quantized(QuantizedB {
                    codes: &code_buffer,
                    scales: &scales,
                    correction: QuantizedCorrection::Symmetric,
                    params: metadata,
                    mode: QuantizationMode::U4,
                    group_size: 32,
                    signed_codes: true,
                }),
                b_leading_dimension: None,
                b_transpose: true,
                output: MatmulOutput {
                    values: &mut output,
                    row_stride: Some(stride),
                    ops: MatmulDOps::none(),
                },
                gather_indices: None::<&Buffer>,
                m,
                n,
                k,
            },
            &mut command,
        )
        .map_err(|e| e.to_string())?;
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = vec![-999.0; (m * stride) as usize];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                let code = ((col * k + inner) * 13 + 7) % 16;
                let scale = scale_values[(col * strides.output_stride + (inner / 32) * strides.group_stride) as usize];
                sum += a_values[(3 + row * k + inner) as usize] * scale * (code as f32 - 8.0);
            }
            expected[(row * stride + col) as usize] = sum;
        }
    }
    compare(&format!("quantized matmul m={m}, {layout:?}"), &output.copyout::<f32>(), &expected, 0.00005)
}

fn full_matmul(
    context: &GpuContext,
    transpose: bool,
) -> Result<Value, String> {
    let (m, n, k) = (17u32, 19u32, 37u32);
    let stride = if transpose {
        k + 3
    } else {
        n + 3
    };
    let av: Vec<_> = (0..m * k).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    let b_rows = if transpose {
        n
    } else {
        k
    };
    let bv: Vec<_> = (0..b_rows * stride).map(|i| (i % 11) as f32 / 16.0 - 0.5).collect();
    let a = buffer_f32(context, &av)?;
    let b = buffer_f32(context, &bv)?;
    let mut d = buffer_f32(context, &vec![0.0; (m * n) as usize])?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    Matmul::new(context, DataType::F32, DataType::F32, DataType::F32)
        .map_err(|e| e.to_string())?
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &b,
                },
                b_leading_dimension: Some(stride),
                b_transpose: transpose,
                output: MatmulOutput::new(&mut d, MatmulDOps::none()),
                gather_indices: None::<&Buffer>,
                m,
                n,
                k,
            },
            &mut command,
        )
        .map_err(|e| e.to_string())?;
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = vec![0.0; (m * n) as usize];
    for row in 0..m {
        for col in 0..n {
            for inner in 0..k {
                let index = if transpose {
                    col * stride + inner
                } else {
                    inner * stride + col
                };
                expected[(row * n + col) as usize] += av[(row * k + inner) as usize] * bv[index as usize];
            }
        }
    }
    compare(&format!("full precision tiled strides transpose={transpose}"), &d.copyout::<f32>(), &expected, 0.00001)
}

fn attention(
    context: &GpuContext,
    dtype: DataType,
    dim: u32,
    suffix: u32,
    prefix: u32,
) -> Result<Value, String> {
    let (heads, kv_heads) = (4u32, 2u32);
    let sequence = suffix + prefix;
    let cv = |v: f32| {
        if dtype == DataType::BF16 {
            bf16(v)
        } else {
            v
        }
    };
    let queries: Vec<_> = (0..heads * suffix * dim).map(|i| cv(((i * 7 % 17) as f32 - 8.0) / 16.0)).collect();
    let keys: Vec<_> = (0..sequence * kv_heads * dim).map(|i| cv(((i * 3 % 13) as f32 - 6.0) / 8.0)).collect();
    let values: Vec<_> = (0..sequence * kv_heads * dim).map(|i| cv(((i * 5 % 11) as f32 - 5.0) / 8.0)).collect();
    let make = |values: &[f32]| {
        if dtype == DataType::BF16 {
            buffer_bf16(context, values)
        } else {
            buffer_f32(context, values)
        }
    };
    let q = make(&queries)?;
    let k = make(&keys)?;
    let v = make(&values)?;
    let mut out = make(&vec![0.0; (heads * suffix * dim) as usize])?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    Attention::new(context, dtype, dim, false, false, true, false, false).map_err(|e| e.to_string())?.encode(
        &q,
        &k,
        &v,
        &mut out,
        heads / kv_heads,
        sequence,
        dim,
        kv_heads * dim,
        dim,
        kv_heads * dim,
        None,
        0.125,
        None::<&Buffer>,
        None,
        None::<&Buffer>,
        heads,
        suffix,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = vec![0.0; (heads * suffix * dim) as usize];
    for query in 0..suffix {
        for head in 0..heads {
            let count = prefix + query + 1;
            let kv = head / (heads / kv_heads);
            let mut logits = vec![0.0; count as usize];
            for key in 0..count {
                for d in 0..dim {
                    logits[key as usize] += queries[((head * suffix + query) * dim + d) as usize]
                        * 0.125
                        * keys[(key * kv_heads * dim + kv * dim + d) as usize];
                }
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities: Vec<_> = logits.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = probabilities.iter().sum();
            for d in 0..dim {
                let mut value = 0.0;
                for key in 0..count {
                    value += probabilities[key as usize] / sum * values[(key * kv_heads * dim + kv * dim + d) as usize];
                }
                expected[((query * heads + head) * dim + d) as usize] = cv(value);
            }
        }
    }
    let actual = if dtype == DataType::BF16 {
        out.copyout::<u16>().iter().map(|&v| f32::from_bits(u32::from(v) << 16)).collect()
    } else {
        out.copyout::<f32>()
    };
    compare(
        &format!("causal GQA {dtype:?} dim={dim} suffix={suffix} prefix={prefix}"),
        &actual,
        &expected,
        if dtype == DataType::BF16 {
            0.002
        } else {
            0.00001
        },
    )
}

fn masked_ring_attention(
    context: &GpuContext,
    dtype: DataType,
    dim: u32,
) -> Result<Value, String> {
    use uzu_engine::backends::common::gpu_types::ring::RingParams;
    let (heads, kv_heads, capacity, filled, offset) = (4u32, 2u32, 38u32, 29u32, 14u32);
    let sequence = capacity + 1;
    let cv = |value| {
        if dtype == DataType::BF16 {
            bf16(value)
        } else {
            value
        }
    };
    let queries: Vec<_> = (0..heads * dim).map(|i| cv((i * 7 % 29) as f32 / 32.0 - 0.375)).collect();
    let mut keys: Vec<_> = (0..sequence * kv_heads * dim).map(|i| cv((i * 13 % 31) as f32 / 32.0 - 0.4)).collect();
    let mut values: Vec<_> = (0..sequence * kv_heads * dim).map(|i| cv((i * 11 % 37) as f32 / 16.0 - 0.75)).collect();
    // Unfilled ring slots must remain masked even when their logical positions
    // fall inside the window. Give them distinct data to expose accidental use.
    for position in filled..capacity {
        let physical = (offset + position) % capacity;
        for i in 0..kv_heads * dim {
            keys[(physical * kv_heads * dim + i) as usize] = cv(1.75);
            values[(physical * kv_heads * dim + i) as usize] = cv(9.0);
        }
    }
    let sink_values: Vec<_> = (0..heads).map(|head| cv(head as f32 - 1.0)).collect();
    let q = float_buffer(context, &queries, dtype)?;
    let k = float_buffer(context, &keys, dtype)?;
    let v = float_buffer(context, &values, dtype)?;
    let sinks = float_buffer(context, &sink_values, dtype)?;
    let mut out = float_buffer(context, &vec![0.0; (heads * dim) as usize], dtype)?;
    let kernel = Attention::new(context, dtype, dim, true, true, true, false, true).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(
        &q,
        &k,
        &v,
        &mut out,
        heads / kv_heads,
        sequence,
        dim,
        kv_heads * dim,
        dim,
        kv_heads * dim,
        Some(RingParams {
            ring_offset: offset,
            ring_length: filled,
        }),
        0.125,
        None::<&Buffer>,
        Some(2),
        Some(&sinks),
        heads,
        1,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    // A two-token causal window contains exactly the last filled ring entry and
    // the new suffix. Both physical indices are even, leaving an empty subgroup
    // even on 128-lane hardware; 32/64-lane hardware has additional empty groups.
    let visible = [(offset + filled - 1) % capacity, capacity];
    let mut expected = vec![0.0; (heads * dim) as usize];
    for head in 0..heads {
        let kv = head / (heads / kv_heads);
        let scores: Vec<f32> = visible
            .iter()
            .map(|&key| {
                (0..dim)
                    .map(|d| {
                        queries[(head * dim + d) as usize]
                            * 0.125
                            * keys[(key * kv_heads * dim + kv * dim + d) as usize]
                    })
                    .sum()
            })
            .collect();
        let maximum = scores.iter().copied().fold(sink_values[head as usize], f32::max);
        let probabilities: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator = probabilities.iter().sum::<f32>() + (sink_values[head as usize] - maximum).exp();
        for d in 0..dim {
            let value = visible
                .iter()
                .zip(&probabilities)
                .map(|(&key, probability)| probability * values[(key * kv_heads * dim + kv * dim + d) as usize])
                .sum::<f32>()
                / denominator;
            expected[(head * dim + d) as usize] = cv(value);
        }
    }
    compare(
        &format!("masked parallel attention ring/sliding/sink {dtype:?} dim={dim}"),
        &floats(&out, dtype),
        &expected,
        if dtype == DataType::BF16 {
            0.004
        } else {
            0.00001
        },
    )
}

fn short_conv(context: &GpuContext) -> Result<Value, String> {
    let (channels, tokens, taps, stride) = (5usize, 3usize, 3usize, 17usize);
    let input: Vec<_> = (0..tokens * stride).map(|i| bf16((i % 13) as f32 / 8.0 - 0.5)).collect();
    let weights: Vec<_> = (0..channels * taps).map(|i| (i % 7) as f32 / 8.0 - 0.25).collect();
    let initial: Vec<_> = (0..channels * 2).map(|i| bf16(i as f32 / 16.0)).collect();
    let input_buffer = buffer_bf16(context, &input)?;
    let weights_buffer = buffer_f32(context, &weights)?;
    let mut state = buffer_bf16(context, &initial)?;
    let mut output = buffer_bf16(context, &vec![0.0; tokens * channels])?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    ShortConv::new(context, DataType::BF16, DataType::F32, false, true).map_err(|e| e.to_string())?.encode(
        &input_buffer,
        &weights_buffer,
        None::<&Buffer>,
        None::<&Buffer>,
        &mut output,
        &mut state,
        tokens as u32,
        taps as u32,
        stride as u32,
        2,
        channels as u32,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected_state = initial;
    let mut expected = vec![0.0; tokens * channels];
    for token in 0..tokens {
        for channel in 0..channels {
            let offset = token * stride + channel;
            let x = input[offset] * input[offset + 2 * channels];
            let acc = weights[channel * taps] * expected_state[channel * 2]
                + weights[channel * taps + 1] * expected_state[channel * 2 + 1]
                + weights[channel * taps + 2] * x;
            expected[token * channels + channel] = bf16(acc * input[offset + channels]);
            expected_state[channel * 2] = expected_state[channel * 2 + 1];
            expected_state[channel * 2 + 1] = bf16(x);
        }
    }
    let decode = |buffer: &Buffer| {
        buffer.copyout::<u16>().iter().map(|&v| f32::from_bits(u32::from(v) << 16)).collect::<Vec<_>>()
    };
    let result = compare("BF16 short-convolution in-place multi-token output", &decode(&output), &expected, 0.005)?;
    compare("BF16 short-convolution packed state", &decode(&state), &expected_state, 0.0)?;
    Ok(result)
}

// Independent Walsh matrix reference (rather than the GPU butterfly algorithm).
fn h32(values: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; values.len()];
    for (block, source) in values.as_chunks::<32>().0.iter().enumerate() {
        for output in 0usize..32 {
            result[block * 32 + output] = source
                .iter()
                .enumerate()
                .map(|(input, value)| {
                    if (input & output).count_ones() % 2 == 0 {
                        *value
                    } else {
                        -*value
                    }
                })
                .sum::<f32>()
                / 32.0f32.sqrt();
        }
    }
    result
}
fn integer_buffer(
    context: &GpuContext,
    values: &[i32],
) -> Result<Buffer, String> {
    let mut buffer = context.create_buffer(values.len() * 4).map_err(|e| e.to_string())?;
    buffer.copyin(values);
    Ok(buffer)
}
fn float_buffer(
    context: &GpuContext,
    values: &[f32],
    dtype: DataType,
) -> Result<Buffer, String> {
    if dtype == DataType::BF16 {
        buffer_bf16(context, values)
    } else {
        buffer_f32(context, values)
    }
}
fn floats(
    buffer: &Buffer,
    dtype: DataType,
) -> Vec<f32> {
    if dtype == DataType::BF16 {
        buffer.copyout::<u16>().into_iter().map(|v| f32::from_bits(u32::from(v) << 16)).collect()
    } else {
        buffer.copyout::<f32>()
    }
}

fn normalization(
    context: &GpuContext,
    residual: bool,
    in_place: bool,
) -> Result<Value, String> {
    let (rows, columns) = (3usize, 352usize);
    let input: Vec<_> = (0..rows * columns).map(|i| bf16((i * 13 % 43) as f32 / 16.0 - 1.25)).collect();
    let shortcut: Vec<_> = (0..rows * columns).map(|i| bf16((i * 7 % 31) as f32 / 16.0 - 0.75)).collect();
    let scale: Vec<_> = (0..columns).map(|i| bf16(0.7 + (i % 7) as f32 / 16.0)).collect();
    let bias: Vec<_> = (0..columns).map(|i| bf16((i % 11) as f32 / 32.0 - 0.15)).collect();
    let factors: Vec<i32> = (0..columns)
        .map(|i| {
            if i % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();
    let epsilon = 0.0003f32;
    let scale_offset = 0.137f32;
    let post = 0.77f32;
    let input_buffer = buffer_bf16(context, &input)?;
    let scales = buffer_bf16(context, &scale)?;
    let biases = buffer_bf16(context, &bias)?;
    let factor_buffer = integer_buffer(context, &factors)?;
    let mut skip = buffer_bf16(context, &shortcut)?;
    let mut output = buffer_bf16(context, &input)?;
    let kernel = Normalization::new(
        context,
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
        DataType::F32,
        in_place,
        residual,
        residual,
        residual,
        residual,
        true,
        residual,
        true,
        true,
        true,
    )
    .map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(
        (!in_place).then_some(&input_buffer),
        Some(&scales),
        Some(&biases),
        &mut output,
        residual.then_some(&mut skip),
        Some(&factor_buffer),
        rows as u32,
        columns as u32,
        epsilon,
        scale_offset,
        post,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = vec![0.0; input.len()];
    let mut expected_skip = shortcut.clone();
    for row in 0..rows {
        let source: Vec<_> = (0..columns)
            .map(|i| {
                let index = row * columns + i;
                let value = if residual {
                    bf16(bf16(input[index] + shortcut[index]) * post)
                } else {
                    input[index]
                };
                if residual {
                    expected_skip[index] = value;
                }
                value
            })
            .collect();
        let mean = if residual {
            source.iter().sum::<f32>() / columns as f32
        } else {
            0.0
        };
        let inverse =
            (source.iter().map(|v| v * v).sum::<f32>() / columns as f32 - mean * mean + epsilon).sqrt().recip();
        let affine: Vec<_> = source
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let normalized = (*value - mean) * inverse;
                let scaled = if residual {
                    bf16(normalized * (scale[i] + scale_offset))
                } else {
                    bf16(bf16(normalized) * bf16(scale[i] + scale_offset))
                };
                bf16(scaled + bias[i]) * factors[i] as f32
            })
            .collect();
        let transformed = h32(&affine);
        for i in 0..columns {
            expected[row * columns + i] = bf16(bf16(transformed[i]) * bf16(post));
        }
    }
    if residual {
        compare("normalization rounded residual shortcut", &floats(&skip, DataType::BF16), &expected_skip, 0.0)?;
    }
    compare(
        &format!("BF16 normalization residual={residual} in_place={in_place} H32"),
        &floats(&output, DataType::BF16),
        &expected,
        0.004,
    )
}

fn normalization_shape_edges(
    context: &GpuContext,
    columns: usize,
    copy: bool,
    residual: bool,
    in_place: bool,
) -> Result<Value, String> {
    let rows = 2;
    let input: Vec<_> = (0..rows * columns).map(|i| bf16((i * 7 % 29) as f32 / 16.0 - 0.75)).collect();
    let shortcut: Vec<_> = (0..rows * columns).map(|i| bf16((i * 11 % 31) as f32 / 16.0 - 0.5)).collect();
    let post = 0.77f32;
    let epsilon = 0.0003f32;
    let input_buffer = buffer_bf16(context, &input)?;
    let mut output = buffer_bf16(context, &input)?;
    let mut skip = buffer_bf16(context, &shortcut)?;
    let kernel = Normalization::new(
        context,
        DataType::BF16,
        DataType::F32,
        DataType::BF16,
        DataType::F32,
        in_place,
        true,
        true,
        copy,
        residual,
        false,
        true,
        false,
        false,
        false,
    )
    .map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(
        (!in_place).then_some(&input_buffer),
        None::<&Buffer>,
        None::<&Buffer>,
        &mut output,
        (copy || residual).then_some(&mut skip),
        None::<&Buffer>,
        rows as u32,
        columns as u32,
        epsilon,
        0.0,
        post,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = Vec::with_capacity(input.len());
    let mut expected_skip = shortcut.clone();
    for row in 0..rows {
        let source: Vec<_> = (0..columns)
            .map(|i| {
                let index = row * columns + i;
                let value = if copy && residual {
                    bf16(bf16(input[index] + shortcut[index]) * post)
                } else {
                    input[index]
                };
                if copy {
                    expected_skip[index] = value;
                }
                value
            })
            .collect();
        let mean = source.iter().sum::<f32>() / columns as f32;
        let inverse =
            (source.iter().map(|v| v * v).sum::<f32>() / columns as f32 - mean * mean + epsilon).sqrt().recip();
        expected.extend(source.iter().enumerate().map(|(i, &value)| {
            // Preserve the existing backend behavior of this unusual flag pair:
            // normalization statistics use input, output uses the old shortcut.
            let value = if residual && !copy {
                shortcut[row * columns + i]
            } else {
                value
            };
            bf16((value - mean) * inverse)
        }));
    }
    compare("normalization shortcut side effects", &floats(&skip, DataType::BF16), &expected_skip, 0.0)?;
    compare(
        &format!("BF16 normalization shape columns={columns} copy={copy} residual={residual} in_place={in_place}"),
        &floats(&output, DataType::BF16),
        &expected,
        0.004,
    )
}

fn embedding(
    context: &GpuContext,
    dtype: DataType,
    quantized: bool,
) -> Result<Value, String> {
    // Three quantization groups per vocabulary row exercise packed zero-point padding.
    let (vocab, columns, group) = (5usize, 96usize, 32usize);
    let ids = [4u32, 1, 9, 0];
    let groups = columns / group;
    let cv = |v| {
        if dtype == DataType::BF16 {
            bf16(v)
        } else {
            v
        }
    };
    let dense: Vec<_> = (0..vocab * columns).map(|i| cv((i * 3 % 29) as f32 / 16.0 - 0.75)).collect();
    let scales: Vec<_> = (0..vocab * groups).map(|i| cv((i % 7 + 1) as f32 / 32.0)).collect();
    let factors: Vec<i32> = (0..columns)
        .map(|i| {
            if i % 3 == 1 {
                -1
            } else {
                1
            }
        })
        .collect();
    let mut codes = vec![0u8; vocab * columns / 2];
    let mut zeros = vec![0u8; vocab * groups.div_ceil(2)];
    for i in 0..vocab * columns {
        codes[i / 2] |= (((i * 7 + 3) % 16) as u8) << ((i % 2) * 4);
    }
    for row in 0..vocab {
        for g in 0..groups {
            zeros[row * groups.div_ceil(2) + g / 2] |= (((row + g + 4) % 16) as u8) << ((g % 2) * 4);
        }
    }
    let values = if quantized {
        let mut b = context.create_buffer(codes.len()).map_err(|e| e.to_string())?;
        b.copyin(&codes);
        b
    } else {
        float_buffer(context, &dense, dtype)?
    };
    let scale_buffer = float_buffer(context, &scales, dtype)?;
    let mut zero_buffer = context.create_buffer(zeros.len()).map_err(|e| e.to_string())?;
    zero_buffer.copyin(&zeros);
    let mut token_ids = context.create_buffer(ids.len() * 4).map_err(|e| e.to_string())?;
    token_ids.copyin(&ids);
    let factor_buffer = integer_buffer(context, &factors)?;
    let mut output = float_buffer(context, &vec![0.0; ids.len() * columns], dtype)?;
    let kernel = Embedding::new(
        context,
        dtype,
        if quantized {
            EmbeddingTableKind::Quantized
        } else {
            EmbeddingTableKind::Dense
        },
        quantized.then_some(group as u32),
        quantized.then_some(QuantizationMode::U4),
        quantized.then_some(QuantizationMethod::ScaleZeroPoint),
        true,
    )
    .map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    let input_scale = 0.73f32;
    kernel.encode(
        &token_ids,
        &values,
        quantized.then_some(&scale_buffer),
        quantized.then_some(&zero_buffer),
        None::<&Buffer>,
        Some(&factor_buffer),
        None::<&Buffer>,
        None::<&Buffer>,
        None::<&Buffer>,
        &mut output,
        ids.len() as u32,
        vocab as u32,
        columns as u32,
        input_scale,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = Vec::new();
    for token in ids {
        let row = token as usize;
        let values: Vec<_> = (0..columns)
            .map(|col| {
                if row >= vocab {
                    0.0
                } else if quantized {
                    let scale = scales[row * groups + col / group];
                    let code = ((row * columns + col) * 7 + 3) % 16;
                    let zero = (row + col / group + 4) % 16;
                    cv((scale * code as f32 - scale * zero as f32) * input_scale)
                } else {
                    cv(dense[row * columns + col] * cv(input_scale))
                }
            })
            .collect();
        expected.extend(h32(&values).iter().enumerate().map(|(i, v)| cv(v * factors[i] as f32)));
    }
    compare(
        &format!("embedding {dtype:?} quantized={quantized} packed-zero H32"),
        &floats(&output, dtype),
        &expected,
        if dtype == DataType::BF16 {
            0.004
        } else {
            0.00001
        },
    )
}

fn activation_transform(
    context: &GpuContext,
    dtype: DataType,
    in_place: bool,
    op: ActivationTransformOp,
) -> Result<Value, String> {
    let (rows, columns) = (3usize, 352usize);
    let cv = |v| {
        if dtype == DataType::BF16 {
            bf16(v)
        } else {
            v
        }
    };
    let input: Vec<_> = (0..rows * columns).map(|i| cv((i * 11 % 47) as f32 / 32.0 - 0.625)).collect();
    let bias: Vec<_> = (0..columns).map(|i| (i % 5) as f32 / 64.0 - 0.04).collect();
    let factors: Vec<i32> = (0..columns)
        .map(|i| {
            if i % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();
    let is_output = op == ActivationTransformOp::OutputRht;
    let with_bias = is_output && in_place;
    let input_buffer = float_buffer(context, &input, dtype)?;
    let mut output = float_buffer(context, &input, dtype)?;
    let bias_buffer = buffer_f32(context, &bias)?;
    let factor_buffer = integer_buffer(context, &factors)?;
    let kernel = Transform::new(context, dtype, DataType::F32, op, false, in_place, 32, 32, with_bias)
        .map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(
        (!in_place).then_some(&input_buffer),
        Some(&mut output),
        with_bias.then_some(&bias_buffer),
        None::<&mut Buffer>,
        None::<&mut Buffer>,
        None::<&mut Buffer>,
        &factor_buffer,
        rows as u32,
        columns as u32,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut expected = Vec::new();
    for source in input.chunks_exact(columns) {
        let multiplied: Vec<_> = source
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if is_output {
                    *v
                } else {
                    v * factors[i] as f32
                }
            })
            .collect();
        expected.extend(h32(&multiplied).iter().enumerate().map(|(i, v)| {
            let value = if is_output {
                v * factors[i] as f32
            } else {
                *v
            };
            if with_bias {
                cv(cv(value) + bias[i])
            } else {
                cv(value)
            }
        }));
    }
    compare(
        &format!("H32 activation {op:?} {dtype:?} in_place={in_place} bias={with_bias}"),
        &floats(&output, dtype),
        &expected,
        if dtype == DataType::BF16 {
            0.004
        } else {
            0.00001
        },
    )
}
