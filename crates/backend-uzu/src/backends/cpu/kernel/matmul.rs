use half::{bf16, f16};

use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
            gpu_types::{HadamardTransformOrder, QuantizationMode},
            kernel::{
                HadamardTransformKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel},
            },
        },
        cpu::{Cpu, buffer::BufferDowncastExt, context::CpuContext, error::CpuError},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct MatmulCpuKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    hadamard: <<Cpu as Backend>::Kernels as Kernels>::HadamardTransformKernel,
}

impl MatmulKernel for MatmulCpuKernel {
    type Backend = Cpu;

    fn new(
        context: &CpuContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, CpuError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Cpu>::UnsupportedDataType(data_type).into());
            }
        }
        let hadamard = <<Cpu as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            output_data_type,
            HadamardTransformOrder::Output,
        )?;
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            hadamard,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Cpu>>>(
        &mut self,
        arguments: MatmulArguments<Cpu, TB>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        let output_scale = arguments.d_transform.ab_scale;
        let accumulate = arguments.d_transform.accumulate;
        let bias_alloc = arguments.d_transform.bias;
        let post_rht = arguments.d_transform.rht_factors;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        let m_u = m as usize;
        let n_u = n as usize;
        let k_u = k as usize;
        let leading_dimension_a = k_u;
        let leading_dimension_d = n_u;
        let weights_data_type = self.weights_data_type;
        let input_data_type = self.input_data_type;
        let output_data_type = self.output_data_type;

        let a_buffer_range = a.as_buffer_range_ref();
        let a_byte_off = a_buffer_range.range().start + a_offset * input_data_type.size_in_bytes();
        let a_ptr = SendPtr(unsafe { &*a_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(a_byte_off));
        let bias_ptr = bias_alloc.map(|bias| {
            let bias_buffer_range = bias.as_buffer_range_ref();
            let bias_range = bias_buffer_range.range();
            SendPtr(unsafe { &*bias_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(bias_range.start))
        });
        let d_ptr = {
            let d_buffer_range = d.as_buffer_range_mut();
            let d_byte_off = d_buffer_range.range().start;
            SendPtrMut(unsafe { (&*d_buffer_range.buffer().get()).as_ptr().wrapping_byte_add(d_byte_off) as *mut u8 })
        };

        let use_zero_point = matches!(b, MatmulB::ScaleZeroPointDequant { .. });

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let leading_dimension_b = b_leading_dimension.map(|ld| ld as usize).unwrap_or(if b_transpose {
                    k_u
                } else {
                    n_u
                });
                let b_buffer_range = weights.as_buffer_range_ref();
                let b_byte_off = b_buffer_range.range().start + b_offset * weights_data_type.size_in_bytes();
                let b_ptr = SendPtr(
                    unsafe { &*b_buffer_range.buffer().downcast().get() }.as_ptr().wrapping_byte_add(b_byte_off),
                );

                let command_buffer = encoder.as_command_buffer_mut();
                command_buffer.push_command(move || {
                    reference_matmul(
                        a_ptr.as_ptr(),
                        input_data_type,
                        ReferenceWeights::FullPrecision {
                            weights: b_ptr.as_ptr(),
                            data_type: weights_data_type,
                            leading_dimension: leading_dimension_b,
                            transpose: b_transpose,
                        },
                        d_ptr.as_ptr(),
                        output_data_type,
                        m_u,
                        n_u,
                        k_u,
                        leading_dimension_a,
                        leading_dimension_d,
                        output_scale,
                        accumulate,
                        bias_ptr.map(|p| p.as_ptr()),
                        weights_data_type,
                    )
                });
            },
            MatmulB::ScaleBiasDequant {
                b: weights,
                scales,
                biases: second,
                mode,
                group_size,
            }
            | MatmulB::ScaleZeroPointDequant {
                b: weights,
                scales,
                zero_points: second,
                mode,
                group_size,
            } => {
                let dequant = if use_zero_point {
                    DequantKind::ZeroPoint
                } else {
                    DequantKind::Bias
                };
                self.encode_quant(
                    encoder,
                    weights,
                    scales,
                    Some(second),
                    dequant,
                    mode,
                    group_size,
                    a_ptr,
                    input_data_type,
                    d_ptr,
                    output_data_type,
                    m_u,
                    n_u,
                    k_u,
                    leading_dimension_a,
                    leading_dimension_d,
                    output_scale,
                    accumulate,
                    bias_ptr,
                    weights_data_type,
                );
            },
            MatmulB::ScaleSymmetricDequant {
                b: weights,
                scales,
                mode,
                group_size,
            } => {
                self.encode_quant(
                    encoder,
                    weights,
                    scales,
                    None,
                    DequantKind::Symmetric,
                    mode,
                    group_size,
                    a_ptr,
                    input_data_type,
                    d_ptr,
                    output_data_type,
                    m_u,
                    n_u,
                    k_u,
                    leading_dimension_a,
                    leading_dimension_d,
                    output_scale,
                    accumulate,
                    bias_ptr,
                    weights_data_type,
                );
            },
        }

        if let Some(factors) = post_rht {
            self.hadamard.encode(d, factors, n, m, encoder);
        }

        Ok(())
    }
}

impl MatmulCpuKernel {
    #[allow(clippy::too_many_arguments)]
    fn encode_quant(
        &self,
        encoder: &mut Encoder<Cpu>,
        weights: &Allocation<Cpu>,
        scales: &Allocation<Cpu>,
        second: Option<&Allocation<Cpu>>,
        dequant: DequantKind,
        mode: QuantizationMode,
        group_size: u32,
        a_ptr: SendPtr<u8>,
        input_data_type: DataType,
        d_ptr: SendPtrMut<u8>,
        output_data_type: DataType,
        m_u: usize,
        n_u: usize,
        k_u: usize,
        leading_dimension_a: usize,
        leading_dimension_d: usize,
        output_scale: f32,
        accumulate: bool,
        bias_ptr: Option<SendPtr<u8>>,
        weights_data_type: DataType,
    ) {
        let bits = match mode {
            QuantizationMode::U4 => 4usize,
            QuantizationMode::I8 | QuantizationMode::U8 => 8usize,
        };
        let group_size = group_size as usize;

        let b_buffer_range = weights.as_buffer_range_ref();
        let weights_ptr = SendPtr(
            unsafe { &*b_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(b_buffer_range.range().start),
        );
        let scales_buffer_range = scales.as_buffer_range_ref();
        let scales_ptr = SendPtr(
            unsafe { &*scales_buffer_range.buffer().get() }
                .as_ptr()
                .wrapping_byte_add(scales_buffer_range.range().start),
        );
        let second_ptr = second.map(|second| {
            let second_buffer_range = second.as_buffer_range_ref();
            SendPtr(
                unsafe { &*second_buffer_range.buffer().get() }
                    .as_ptr()
                    .wrapping_byte_add(second_buffer_range.range().start),
            )
        });

        let command_buffer = encoder.as_command_buffer_mut();
        command_buffer.push_command(move || {
            reference_matmul(
                a_ptr.as_ptr(),
                input_data_type,
                ReferenceWeights::Quantized(ReferenceQuantWeights {
                    weights: weights_ptr.as_ptr() as *const u32,
                    scales: scales_ptr.as_ptr(),
                    scales_data_type: weights_data_type,
                    zero_points: match dequant {
                        DequantKind::ZeroPoint => second_ptr.map(|p| p.as_ptr()),
                        _ => None,
                    },
                    biases: match dequant {
                        DequantKind::Bias => second_ptr.map(|p| p.as_ptr()),
                        _ => None,
                    },
                    symmetric: matches!(dequant, DequantKind::Symmetric),
                    bits,
                    group_size,
                }),
                d_ptr.as_ptr(),
                output_data_type,
                m_u,
                n_u,
                k_u,
                leading_dimension_a,
                leading_dimension_d,
                output_scale,
                accumulate,
                bias_ptr.map(|p| p.as_ptr()),
                weights_data_type,
            )
        });
    }
}

#[derive(Clone, Copy)]
enum DequantKind {
    ZeroPoint,
    Bias,
    Symmetric,
}

/// Group-quantized weight operand (weights laid out transposed as [N, K]).
struct ReferenceQuantWeights {
    weights: *const u32,
    scales: *const u8,
    scales_data_type: DataType,
    /// At most one of `zero_points` / `biases` is set, selecting the
    /// scale+zero-point vs scale+bias dequantization. When `symmetric` is set,
    /// neither is present and the offset is the implicit midpoint.
    zero_points: Option<*const u8>,
    biases: Option<*const u8>,
    symmetric: bool,
    bits: usize,
    group_size: usize,
}

/// Weight operand for the reference matmul: either full precision or quantized.
enum ReferenceWeights {
    FullPrecision {
        weights: *const u8,
        data_type: DataType,
        /// Leading dimension (row/col stride) of the weight matrix.
        leading_dimension: usize,
        /// `true` when weights are [N, K] (the GEMV / quantized layout),
        /// `false` when they are [K, N].
        transpose: bool,
    },
    Quantized(ReferenceQuantWeights),
}

#[inline]
unsafe fn read_f32(
    base: *const u8,
    data_type: DataType,
    index: usize,
) -> f32 {
    unsafe {
        match data_type {
            DataType::F32 => *(base as *const f32).add(index),
            DataType::F16 => (*(base as *const f16).add(index)).to_f32(),
            DataType::BF16 => (*(base as *const bf16).add(index)).to_f32(),
            _ => unreachable!(),
        }
    }
}

#[inline]
unsafe fn write_f32(
    base: *mut u8,
    data_type: DataType,
    index: usize,
    value: f32,
) {
    unsafe {
        match data_type {
            DataType::F32 => *(base as *mut f32).add(index) = value,
            DataType::F16 => *(base as *mut f16).add(index) = f16::from_f32(value),
            DataType::BF16 => *(base as *mut bf16).add(index) = bf16::from_f32(value),
            _ => unreachable!(),
        }
    }
}

/// Single fully-featured CPU reference matmul, used purely for unit-test
/// correctness (performance is irrelevant). Each operand is read with its own
/// data type and accumulated in f32, so it covers mixed-precision combos. For
/// every (i, j) it computes:
///
///   D[i, j] = output_scale * sum_l A[i, l] * Bval[j, l]
///             + (accumulate ? D[i, j] : 0)
///             + (bias       ? bias[j]  : 0)
///
/// where `Bval` is a full-precision weight or a dequantized group-quantized
/// weight. Output transforms beyond scale/accumulate/bias (e.g. RHT) are
/// applied by the caller as a separate pass.
#[allow(clippy::too_many_arguments)]
fn reference_matmul(
    a: *const u8,
    a_data_type: DataType,
    weights: ReferenceWeights,
    output: *mut u8,
    output_data_type: DataType,
    m: usize,
    n: usize,
    k: usize,
    leading_dimension_a: usize,
    leading_dimension_d: usize,
    output_scale: f32,
    accumulate: bool,
    bias: Option<*const u8>,
    bias_data_type: DataType,
) {
    let quant_layout = match &weights {
        ReferenceWeights::Quantized(q) => {
            let num_groups_k = k.div_ceil(q.group_size);
            let zero_point_stride = if q.bits == 4 {
                num_groups_k.div_ceil(2)
            } else {
                num_groups_k
            };
            let pack_factor = if q.bits == 4 {
                8
            } else {
                4
            };
            Some((num_groups_k, zero_point_stride, pack_factor))
        },
        ReferenceWeights::FullPrecision {
            ..
        } => None,
    };

    unsafe {
        for row in 0..m {
            for col in 0..n {
                let mut accumulator = 0.0f32;
                for inner in 0..k {
                    let a_value = read_f32(a, a_data_type, row * leading_dimension_a + inner);
                    let b_value = match &weights {
                        ReferenceWeights::FullPrecision {
                            weights,
                            data_type,
                            leading_dimension,
                            transpose,
                        } => {
                            let index = if *transpose {
                                col * leading_dimension + inner
                            } else {
                                inner * leading_dimension + col
                            };
                            read_f32(*weights, *data_type, index)
                        },
                        ReferenceWeights::Quantized(q) => {
                            let (num_groups_k, zero_point_stride, pack_factor) = quant_layout.unwrap();
                            let weight_linear_index = col * k + inner;
                            let quantized_value = if q.bits == 4 {
                                let word_index = weight_linear_index / pack_factor;
                                let bit_offset = (weight_linear_index % pack_factor) * 4;
                                ((q.weights.add(word_index).read_unaligned() >> bit_offset) & 0xF) as f32
                            } else {
                                let word_index = weight_linear_index / pack_factor;
                                let bit_offset = (weight_linear_index % pack_factor) * 8;
                                ((q.weights.add(word_index).read_unaligned() >> bit_offset) & 0xFF) as f32
                            };
                            let group_index = inner / q.group_size;
                            let scale = read_f32(q.scales, q.scales_data_type, col * num_groups_k + group_index);
                            let bias_term = if let Some(zero_points) = q.zero_points {
                                let zero_point = if q.bits == 4 {
                                    let byte_index = col * zero_point_stride + (group_index >> 1);
                                    let byte_value = *zero_points.add(byte_index);
                                    if (group_index & 1) == 0 {
                                        (byte_value & 0x0F) as f32
                                    } else {
                                        ((byte_value >> 4) & 0x0F) as f32
                                    }
                                } else {
                                    *zero_points.add(col * zero_point_stride + group_index) as f32
                                };
                                -scale * zero_point
                            } else if q.symmetric {
                                let midpoint = (1u32 << (q.bits - 1)) as f32;
                                -scale * midpoint
                            } else {
                                read_f32(q.biases.unwrap(), q.scales_data_type, col * num_groups_k + group_index)
                            };
                            scale * quantized_value + bias_term
                        },
                    };
                    accumulator += a_value * b_value;
                }

                let output_index = row * leading_dimension_d + col;
                let mut value = output_scale * accumulator;
                if accumulate {
                    value += read_f32(output, output_data_type, output_index);
                }
                if let Some(bias) = bias {
                    value += read_f32(bias, bias_data_type, col);
                }
                write_f32(output, output_data_type, output_index, value);
            }
        }
    }
}
