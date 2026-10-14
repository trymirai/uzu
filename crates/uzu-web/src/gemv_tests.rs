//! Quantized decode matrix-vector fixtures shared by subgroup and fallback runs.
use serde_json::{Value, json};
use uzu_engine::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::QuantizationMode,
            kernel::matmul::{
                Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput, QuantParams,
                QuantParamsLayout, QuantizedB, QuantizedCorrection,
            },
        },
        webgpu::WebGpu,
    },
    data_type::DataType,
};

type GpuContext = <WebGpu as Backend>::Context;
type Buffer = <WebGpu as Backend>::GlobalBuffer;
type Matmul = <<WebGpu as Backend>::Kernels as Kernels>::MatmulKernel;

#[derive(Clone, Copy, Debug)]
enum Correction {
    Symmetric,
    Biases,
    ZeroPoints,
}

struct Case {
    bits: u32,
    layout: QuantParamsLayout,
    correction: Correction,
    signed: bool,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    gather: bool,
    output_bf16: bool,
    int8_layout: Option<Int8CodeLayout>,
    soft_cap: bool,
    input_offset: u32,
}

pub(crate) fn run(context: &GpuContext) -> Result<Value, String> {
    let mut cases = Vec::new();
    for bits in [4, 8] {
        for layout in [QuantParamsLayout::OutputGroup, QuantParamsLayout::GroupOutput] {
            for (index, correction) in
                [Correction::Symmetric, Correction::Biases, Correction::ZeroPoints].into_iter().enumerate()
            {
                cases.push(check(
                    context,
                    Case {
                        bits,
                        layout,
                        correction,
                        signed: index != 1,
                        m: index as u32 + 1,
                        n: 19,
                        k: 1056,
                        group_size: 32,
                        gather: false,
                        output_bf16: index == 2,
                        int8_layout: None,
                        soft_cap: false,
                        input_offset: 3,
                    },
                )?);
            }
        }
    }
    for int8_layout in [None, Some(Int8CodeLayout::Sequential), Some(Int8CodeLayout::GroupedByNibble)] {
        cases.push(check(
            context,
            Case {
                bits: 4,
                layout: QuantParamsLayout::GroupOutput,
                correction: Correction::ZeroPoints,
                signed: false,
                m: 7,
                n: 23,
                k: 1056,
                group_size: 32,
                gather: true,
                output_bf16: false,
                int8_layout,
                soft_cap: false,
                input_offset: 3,
            },
        )?);
    }
    // Quantization groups crossing packed words must retain the generic path.
    cases.push(check(
        context,
        Case {
            bits: 4,
            layout: QuantParamsLayout::OutputGroup,
            correction: Correction::Biases,
            signed: true,
            m: 4,
            n: 17,
            k: 66,
            group_size: 3,
            gather: false,
            output_bf16: false,
            int8_layout: None,
            soft_cap: true,
            input_offset: 3,
        },
    )?);
    // Fewer packed words than subgroup lanes, and a single partial output tile.
    cases.push(check(
        context,
        Case {
            bits: 8,
            layout: QuantParamsLayout::GroupOutput,
            correction: Correction::Symmetric,
            signed: true,
            m: 1,
            n: 1,
            k: 32,
            group_size: 32,
            gather: false,
            output_bf16: true,
            int8_layout: None,
            soft_cap: false,
            input_offset: 3,
        },
    )?);
    // Exercise vectorized BF16 reads as well as the unaligned views above.
    // Odd quantization-group counts and output widths test both tail paths.
    for group_size in [32, 64] {
        for correction in [Correction::Symmetric, Correction::Biases, Correction::ZeroPoints] {
            cases.push(check(
                context,
                Case {
                    bits: 4,
                    layout: QuantParamsLayout::GroupOutput,
                    correction,
                    signed: true,
                    m: 2,
                    n: 17,
                    k: group_size * 17,
                    group_size,
                    gather: false,
                    output_bf16: false,
                    int8_layout: None,
                    soft_cap: false,
                    input_offset: 4,
                },
            )?);
        }
    }
    // Prefill must transpose contiguous packed-weight loads into the shared
    // tile while preserving partial M/N/K tiles and both metadata layouts.
    for (bits, layout, correction) in [
        (4, QuantParamsLayout::OutputGroup, Correction::Biases),
        (8, QuantParamsLayout::GroupOutput, Correction::ZeroPoints),
    ] {
        cases.push(check(
            context,
            Case {
                bits,
                layout,
                correction,
                signed: true,
                m: 17,
                n: 19,
                k: 66,
                group_size: 3,
                gather: false,
                output_bf16: false,
                int8_layout: None,
                soft_cap: false,
                input_offset: 3,
            },
        )?);
    }
    for output_bf16 in [false, true] {
        for correction in [Correction::Symmetric, Correction::Biases, Correction::ZeroPoints] {
            cases.push(check_with_rht(
                context,
                Case {
                    bits: 4,
                    layout: QuantParamsLayout::GroupOutput,
                    correction,
                    signed: true,
                    m: 2,
                    n: 96,
                    k: 1088,
                    group_size: 64,
                    gather: matches!(correction, Correction::ZeroPoints),
                    output_bf16,
                    int8_layout: None,
                    soft_cap: !output_bf16,
                    input_offset: 4,
                },
                true,
            )?);
        }
    }
    // Short prompts straddle the row-parallel/tiled boundary. Keep output
    // padding intact and preserve BF16 rounding around the fused transform.
    for m in [5, 17, 33] {
        for output_rht in [false, true] {
            let tiled_tails = m == 33 && !output_rht;
            cases.push(check_with_rht(
                context,
                Case {
                    bits: 4,
                    layout: QuantParamsLayout::OutputGroup,
                    correction: Correction::Symmetric,
                    signed: true,
                    m,
                    n: if output_rht {
                        96
                    } else if tiled_tails {
                        65
                    } else {
                        131
                    },
                    k: if tiled_tails {
                        66
                    } else {
                        1088
                    },
                    group_size: if tiled_tails {
                        3
                    } else {
                        64
                    },
                    gather: false,
                    output_bf16: true,
                    int8_layout: None,
                    soft_cap: false,
                    input_offset: 4,
                },
                output_rht,
            )?);
        }
    }
    // Wide short prompts select the four-row prefill path before M=33.
    // Keep its partial final row tile and BF16 rounding around output RHT
    // covered at the actual 2048-column/2048-K routing boundary.
    cases.push(check_with_rht(
        context,
        Case {
            bits: 4,
            layout: QuantParamsLayout::GroupOutput,
            correction: Correction::Symmetric,
            signed: true,
            m: 17,
            n: 2048,
            k: 2048,
            group_size: 64,
            gather: false,
            output_bf16: true,
            int8_layout: None,
            soft_cap: false,
            input_offset: 4,
        },
        true,
    )?);
    // Packed W4 prefill reuses weights across four rows. Exercise incomplete
    // row/column tiles, both group sizes/layouts and every correction mode.
    for group_size in [32, 64] {
        for layout in [QuantParamsLayout::OutputGroup, QuantParamsLayout::GroupOutput] {
            for (index, correction) in
                [Correction::Symmetric, Correction::Biases, Correction::ZeroPoints].into_iter().enumerate()
            {
                cases.push(check_with_rht(
                    context,
                    Case {
                        bits: 4,
                        layout,
                        correction,
                        signed: index != 1,
                        m: if index == 1 {
                            65
                        } else {
                            33
                        },
                        n: if index == 2 {
                            96
                        } else {
                            35
                        },
                        k: group_size * 17,
                        group_size,
                        gather: false,
                        output_bf16: index != 1,
                        int8_layout: None,
                        soft_cap: index == 1,
                        input_offset: if group_size == 32 {
                            3
                        } else {
                            4
                        },
                    },
                    index == 2,
                )?);
            }
        }
    }
    for group_size in [32, 64] {
        cases.push(check_with_input_type(
            context,
            Case {
                bits: 4,
                layout: QuantParamsLayout::GroupOutput,
                correction: Correction::Symmetric,
                signed: false,
                m: 33,
                n: 35,
                k: group_size * 17,
                group_size,
                gather: false,
                output_bf16: false,
                int8_layout: None,
                soft_cap: false,
                input_offset: 3,
            },
            false,
            false,
        )?);
    }
    cases.push(prefill_f16_offsets(context)?);
    for axis in ["columns", "rows"] {
        cases.push(prefill_large_grid(context, axis)?);
    }
    cases.push(grouped_activation_alignment(context)?);
    Ok(json!({
        "subgroupFastPath": context.supports_subgroups() && context.subgroup_id_supported(),
        "subgroupSizeRange": context.subgroup_size_range(), "cases": cases,
    }))
}

fn half_words(
    context: &GpuContext,
    values: &[u16],
) -> Result<Buffer, String> {
    let mut buffer = context.create_buffer(values.len() * 2).map_err(|e| e.to_string())?;
    buffer.copyin(values);
    Ok(buffer)
}

fn prefill_f16_offsets(context: &GpuContext) -> Result<Value, String> {
    // Offset all three storage types, including halfword-aligned input, scale
    // and output views. Odd output strides share a word with padding values.
    let m = 33;
    let k = 32;
    let stride = 3;
    let guard = 0x3555u16;
    let mut input = vec![guard; 3 + m * k + 1];
    for row in 0..m {
        input[3 + row * k..3 + (row + 1) * k].fill([0x3c00, 0x4000, 0x4200, 0x4400][row % 4]);
    }
    let a = half_words(context, &input)?;
    let scales = half_words(context, &[guard, 0x3800, guard])?;
    let codes = words(context, &[0xdeadbeef, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xdeadbeef])?;
    let bias = half_words(context, &[0x3800])?;
    let mut expected = vec![guard; 1 + m * stride + 1];
    let mut output = half_words(context, &expected)?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    Matmul::new(context, DataType::F16, DataType::F16, DataType::F16)
        .map_err(|e| e.to_string())?
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a,
                    offset: 3,
                },
                b: MatmulB::Quantized(QuantizedB {
                    codes: (&codes).subrange(4..20),
                    scales: (&scales).subrange(2..4),
                    params: QuantParams::new(QuantParamsLayout::OutputGroup, 1, 1),
                    mode: QuantizationMode::U4,
                    group_size: 32,
                    signed_codes: false,
                    correction: QuantizedCorrection::Symmetric,
                }),
                b_transpose: true,
                b_leading_dimension: None,
                output: MatmulOutput {
                    values: (&mut output).subrange_mut(2..2 * (1 + m * stride)),
                    row_stride: Some(stride as u32),
                    ops: MatmulDOps {
                        ab_scale: 0.5,
                        bias: Some(&bias),
                        rht_factors: None,
                        soft_cap: None,
                    },
                },
                gather_indices: None::<&Buffer>,
                m: m as u32,
                n: 1,
                k: k as u32,
            },
            &mut command,
        )
        .map_err(|e| e.to_string())?;
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    for row in 0..m {
        expected[1 + row * stride] = [0x4c20, 0x5010, 0x5210, 0x5408][row % 4];
    }
    let actual = output.copyout::<u16>();
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        if actual != expected {
            return Err(format!("W4 prefill F16 offsets: element {index}: {actual:#06x} != {expected:#06x}"));
        }
    }
    Ok(json!({"name":"W4 prefill F16 input/metadata/output halfword offsets", "values":actual.len()}))
}

fn prefill_large_grid(
    context: &GpuContext,
    axis: &str,
) -> Result<Value, String> {
    let name = format!("W4 prefill paged {axis} grid");
    if !(context.supports_subgroups()
        && context.subgroup_id_supported()
        && context.subgroup_size_range() == Some((32, 32)))
    {
        return Ok(json!({"name":name,"skipped":"requires the fixed-32 subgroup prefill route"}));
    }
    let (m, n) = if axis == "columns" {
        (33usize, 16 * 65535 + 1)
    } else {
        (4 * 65535 + 5, 1usize)
    };
    let k = 32;
    let stride = n + 2;
    let input: Vec<_> = (0..m * k).map(|i| bf16_bits((1 + (i / k) % 7) as f32)).collect();
    let a = half_words(context, &input)?;
    let codes: Vec<_> = (0..n * 4).map(|i| (9 + (i / 4) as u32 % 3) * 0x11111111).collect();
    let codes = words(context, &codes)?;
    let scales = half_words(context, &vec![bf16_bits(1.0); n])?;
    // WebGPU initializes these bytes to zero. Read back only selected values;
    // a full CPU mirror would obscure this dispatch-boundary regression test.
    let mut output = context.create_buffer(2 * (1 + m * stride + 1)).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    Matmul::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
        .map_err(|e| e.to_string())?
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a,
                    offset: 0,
                },
                b: MatmulB::Quantized(QuantizedB {
                    codes: &codes,
                    scales: &scales,
                    params: QuantParams::new(QuantParamsLayout::OutputGroup, n as u32, 1),
                    mode: QuantizationMode::U4,
                    group_size: 32,
                    signed_codes: false,
                    correction: QuantizedCorrection::Symmetric,
                }),
                b_transpose: true,
                b_leading_dimension: None,
                output: MatmulOutput {
                    values: (&mut output).subrange_mut(2..2 * (1 + m * stride)),
                    row_stride: Some(stride as u32),
                    ops: MatmulDOps {
                        ab_scale: 1.0,
                        bias: None,
                        rht_factors: None,
                        soft_cap: None,
                    },
                },
                gather_indices: None::<&Buffer>,
                m: m as u32,
                n: n as u32,
                k: k as u32,
            },
            &mut command,
        )
        .map_err(|e| e.to_string())?;
    let rows = if axis == "columns" {
        vec![0, 16, m - 1]
    } else {
        vec![0, 4 * 65535 - 1, 4 * 65535, m - 1]
    };
    let columns = if axis == "columns" {
        vec![0, 16 * 65535 - 1, n - 1]
    } else {
        vec![0]
    };
    let mut probes = vec![(0, 0), (1 + m * stride, 0)];
    for row in rows {
        for &column in &columns {
            let expected = bf16_bits((32 * (1 + row % 7) * (1 + column % 3)) as f32);
            probes.push((1 + row * stride + column, expected));
        }
        probes.push((1 + row * stride + n, 0));
        probes.push((1 + row * stride + n + 1, 0));
    }
    let mut readback = context.create_buffer(probes.len() * 2).map_err(|e| e.to_string())?;
    for (index, &(element, _)) in probes.iter().enumerate() {
        command.encode_copy(
            (&output).subrange(element * 2..(element + 1) * 2),
            (&mut readback).subrange_mut(index * 2..(index + 1) * 2),
        );
    }
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let actual = readback.copyout::<u16>();
    for (&value, &(element, expected)) in actual.iter().zip(&probes) {
        if value != expected {
            return Err(format!("{name}: element {element}: {value:#06x} != {expected:#06x}"));
        }
    }
    Ok(json!({"name":name,"m":m,"n":n,"k":k,"probes":probes.len()}))
}

fn grouped_activation_alignment(context: &GpuContext) -> Result<Value, String> {
    // Sequential int8 rows may have an arbitrary scale-aligned K. The grouped
    // layout instead packs across eight codes, so a row may not split a block.
    let k = 66;
    let mut values = context.create_buffer(2 * k).map_err(|e| e.to_string())?;
    values.copyin(&vec![1i8; 2 * k]);
    let scales = floats(context, &vec![1.0; 2 * k / 3], false)?;
    let weights = floats(context, &vec![1.0; k], true)?;
    let mut output = floats(context, &[0.0, 0.0], false)?;
    let mut matmul = Matmul::new(context, DataType::BF16, DataType::BF16, DataType::F32).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    for code_layout in [Int8CodeLayout::GroupedByNibble, Int8CodeLayout::Sequential] {
        let result = matmul.encode(
            MatmulArguments {
                a: MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
                    scale_group_size: 3,
                    group_sums: None,
                    code_layout,
                },
                b: MatmulB::FullPrecision {
                    b: &weights,
                },
                b_transpose: true,
                b_leading_dimension: None,
                output: MatmulOutput {
                    values: &mut output,
                    row_stride: None,
                    ops: MatmulDOps {
                        ab_scale: 1.0,
                        bias: None,
                        rht_factors: None,
                        soft_cap: None,
                    },
                },
                gather_indices: None::<&Buffer>,
                m: 2,
                n: 1,
                k: k as u32,
            },
            &mut command,
        );
        if code_layout.is_grouped_by_nibble() {
            if result.is_ok() {
                return Err("Grouped int8 activation rows accepted K=66".into());
            }
        } else {
            result.map_err(|e| e.to_string())?;
        }
    }
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let actual = output.copyout::<f32>();
    if actual != [66.0, 66.0] {
        return Err(format!("Sequential int8 K=66/group=3: {actual:?}"));
    }
    Ok(json!({"name":"grouped int8 row alignment rejection / sequential non-aligned K", "values":2}))
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
fn bf16(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(value)) << 16)
}
fn words(
    context: &GpuContext,
    values: &[u32],
) -> Result<Buffer, String> {
    let mut buffer = context.create_buffer(values.len() * 4).map_err(|e| e.to_string())?;
    buffer.copyin(values);
    Ok(buffer)
}
fn floats(
    context: &GpuContext,
    values: &[f32],
    as_bf16: bool,
) -> Result<Buffer, String> {
    let mut buffer = context
        .create_buffer(
            values.len()
                * if as_bf16 {
                    2
                } else {
                    4
                },
        )
        .map_err(|e| e.to_string())?;
    if as_bf16 {
        buffer.copyin(&values.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>());
    } else {
        buffer.copyin(values);
    }
    Ok(buffer)
}

fn check(
    context: &GpuContext,
    case: Case,
) -> Result<Value, String> {
    check_with_rht(context, case, false)
}

fn check_with_rht(
    context: &GpuContext,
    case: Case,
    output_rht: bool,
) -> Result<Value, String> {
    check_with_input_type(context, case, output_rht, true)
}

fn check_with_input_type(
    context: &GpuContext,
    case: Case,
    output_rht: bool,
    input_bf16: bool,
) -> Result<Value, String> {
    let Case {
        bits,
        layout,
        correction,
        signed,
        m,
        n,
        k,
        group_size,
        gather,
        output_bf16,
        int8_layout,
        soft_cap,
        input_offset,
    } = case;
    let packed = 32 / bits;
    let mask = (1 << bits) - 1;
    let columns = if gather {
        n + 6
    } else {
        n
    };
    let groups = k / group_size;
    let metadata = QuantParams::new(layout, columns, groups);
    let strides = metadata.scale_strides();
    let shape = metadata.scale_shape();
    let mut scales = vec![0.0; (shape[0] * shape[1]) as usize];
    let mut biases = scales.clone();
    let mode = if bits == 4 {
        QuantizationMode::U4
    } else {
        QuantizationMode::U8
    };
    let zp_strides = metadata.zero_point_strides(mode);
    let zp_count = (columns - 1) * zp_strides.output_stride + (groups - 1) * zp_strides.group_stride + 1;
    let mut zero_points = vec![0; zp_count.div_ceil(packed) as usize];
    for column in 0..columns {
        for group in 0..groups {
            let index = (column * strides.output_stride + group * strides.group_stride) as usize;
            scales[index] = (1 + (column + group * 3) % 4) as f32 / 128.0;
            biases[index] = ((column + group) % 7) as f32 / 32.0 - 0.125;
            let point = (column * 7 + group * 3) & mask;
            let point_index = column * zp_strides.output_stride + group * zp_strides.group_stride;
            zero_points[(point_index / packed) as usize] |= point << ((point_index % packed) * bits);
        }
    }
    let scale_buffer = floats(context, &scales, true)?;
    let quant_bias_buffer = floats(context, &biases, true)?;
    let zero_point_buffer = words(context, &zero_points)?;
    let mut codes = vec![0; (columns * k).div_ceil(packed) as usize];
    for index in 0..columns * k {
        let code = (index * 13 + index / 17 + 7) & mask;
        let stored = if signed {
            code ^ (1 << (bits - 1))
        } else {
            code
        };
        codes[(index / packed) as usize] |= stored << ((index % packed) * bits);
    }
    let code_buffer = words(context, &codes)?;
    let offset = if int8_layout.is_some() {
        0
    } else {
        input_offset
    };
    let mut av: Vec<_> = (0..m * k + offset).map(|i| ((i * 7 % 29) as f32 - 14.0) / 32.0).collect();
    let mut activation_codes = vec![0i8; (m * k) as usize];
    let activation_scales: Vec<_> = (0..m * k / 32).map(|i| (1 + i % 4) as f32 / 64.0).collect();
    if let Some(code_layout) = int8_layout {
        for index in 0..m * k {
            let code = ((index * 11 % 31) as i32 - 15) as i8;
            activation_codes[code_layout.index(index as usize)] = code;
            av[index as usize] = f32::from(code) * activation_scales[(index / 32) as usize];
        }
    }
    let mut activation_buffer = context.create_buffer(activation_codes.len()).map_err(|e| e.to_string())?;
    activation_buffer.copyin(&activation_codes);
    let activation_scale_buffer = floats(context, &activation_scales, false)?;
    let a_buffer = floats(context, &av, input_bf16)?;
    let gather_values: Vec<_> = (0..m * n).map(|i| (i * 7 + 3) % columns).collect();
    let gather_buffer = words(context, &gather_values)?;
    let output_bias: Vec<_> = (0..n).map(|i| (i % 5) as f32 / 16.0 - 0.125).collect();
    let output_bias_buffer = floats(context, &output_bias, true)?;
    let factors: Vec<i32> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();
    let factor_buffer = words(context, &factors.iter().map(|&v| v as u32).collect::<Vec<_>>())?;
    let stride = if output_rht {
        n
    } else {
        n + 2
    };
    let mut expected = vec![-999.0; (m * stride) as usize];
    let mut output = floats(context, &expected, output_bf16)?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    let a = if let Some(code_layout) = int8_layout {
        MatmulA::Int8Symmetric {
            values: &activation_buffer,
            scales: &activation_scale_buffer,
            scale_group_size: 32,
            group_sums: None,
            code_layout,
        }
    } else {
        MatmulA::FullPrecision {
            values: &a_buffer,
            offset: offset as usize,
        }
    };
    Matmul::new(
        context,
        DataType::BF16,
        if input_bf16 {
            DataType::BF16
        } else {
            DataType::F32
        },
        if output_bf16 {
            DataType::BF16
        } else {
            DataType::F32
        },
    )
    .map_err(|e| e.to_string())?
    .encode(
        MatmulArguments {
            a,
            b: MatmulB::Quantized(QuantizedB {
                codes: &code_buffer,
                scales: &scale_buffer,
                params: metadata,
                mode,
                group_size,
                signed_codes: signed,
                correction: match correction {
                    Correction::Symmetric => QuantizedCorrection::Symmetric,
                    Correction::Biases => QuantizedCorrection::Biases(&quant_bias_buffer),
                    Correction::ZeroPoints => QuantizedCorrection::ZeroPoints(&zero_point_buffer),
                },
            }),
            b_transpose: true,
            b_leading_dimension: None,
            output: MatmulOutput {
                values: &mut output,
                row_stride: Some(stride),
                ops: MatmulDOps {
                    ab_scale: 0.5,
                    bias: Some(&output_bias_buffer),
                    rht_factors: output_rht.then_some(&factor_buffer),
                    soft_cap: soft_cap.then_some(1.75),
                },
            },
            gather_indices: gather.then_some(&gather_buffer),
            m,
            n,
            k,
        },
        &mut command,
    )
    .map_err(|e| e.to_string())?;
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    for row in 0..m {
        for column in 0..n {
            let weight_column = if gather {
                gather_values[(row * n + column) as usize]
            } else {
                column
            };
            let mut sum = 0.0;
            for inner in 0..k {
                let index = weight_column * k + inner;
                let code = (index * 13 + index / 17 + 7) & mask;
                let group = inner / group_size;
                let metadata_index = (weight_column * strides.output_stride + group * strides.group_stride) as usize;
                let scale = scales[metadata_index];
                let correction = match correction {
                    Correction::Symmetric => -scale * (1 << (bits - 1)) as f32,
                    Correction::Biases => biases[metadata_index],
                    Correction::ZeroPoints => -scale * ((weight_column * 7 + group * 3) & mask) as f32,
                };
                sum += av[(offset + row * k + inner) as usize] * (scale * code as f32 + correction);
            }
            let mut value = sum * 0.5;
            if !output_rht {
                value += output_bias[column as usize];
            }
            if soft_cap {
                value = 1.75 * (value / 1.75).tanh();
            }
            expected[(row * stride + column) as usize] = value;
        }
    }
    if output_rht {
        for row in expected.chunks_exact_mut(n as usize) {
            for (block_index, block) in row.chunks_exact_mut(32).enumerate() {
                if output_bf16 {
                    for value in block.iter_mut() {
                        *value = bf16(*value);
                    }
                }
                for step in [1, 2, 4, 8, 16] {
                    for base in (0..32).step_by(step * 2) {
                        for inner in 0..step {
                            let a = block[base + inner];
                            let b = block[base + inner + step];
                            block[base + inner] = a + b;
                            block[base + inner + step] = a - b;
                        }
                    }
                }
                for (index, value) in block.iter_mut().enumerate() {
                    let column = block_index * 32 + index;
                    *value *= (1.0 / 32.0f32.sqrt()) * factors[column] as f32;
                    if output_bf16 {
                        *value = bf16(*value);
                    }
                    *value += output_bias[column];
                }
            }
        }
    }
    let actual = if output_bf16 {
        for value in &mut expected {
            *value = bf16(*value);
        }
        output.copyout::<u16>().iter().map(|&v| f32::from_bits(u32::from(v) << 16)).collect::<Vec<_>>()
    } else {
        output.copyout::<f32>()
    };
    let name = format!(
        "U{bits} {correction:?} {layout:?} signed={signed} m={m} n={n} k={k} gather={gather} BF16={output_bf16} inputBF16={input_bf16} int8={} softcap={soft_cap} rht={output_rht}",
        int8_layout.is_some()
    );
    let mut max_error = 0.0f32;
    for (index, (&value, &reference)) in actual.iter().zip(&expected).enumerate() {
        let error = (value - reference).abs();
        max_error = max_error.max(error);
        let tolerance = if soft_cap || output_rht && !output_bf16 {
            0.00001 * (1.0 + reference.abs())
        } else {
            0.0
        };
        if !value.is_finite() || error > tolerance {
            return Err(format!("{name}: at {index} actual {value} expected {reference}"));
        }
    }
    Ok(json!({"name":name,"maxAbsoluteError":max_error,"values":actual.len()}))
}
