use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement, DataType,
    backends::{
        common::{
            AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
            gpu_types::{HadamardTransformOrder, QuantizationMode},
            kernel::{
                HadamardTransformKernel, ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel},
            },
        },
        cpu::{BufferDowncastExt, Cpu, context::CpuContext, error::CpuError, kernel::CpuKernels},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

impl ManualKernels for CpuKernels {
    type MatmulKernel = MatmulCpuKernel;
}

pub struct MatmulCpuKernel {
    data_type: DataType,
    hadamard: <<Cpu as Backend>::Kernels as Kernels>::HadamardTransformKernel,
}

impl MatmulKernel for MatmulCpuKernel {
    type Backend = Cpu;

    fn new(
        context: &CpuContext,
        data_type: DataType,
    ) -> Result<Self, CpuError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::<Cpu>::UnsupportedDataType(data_type).into());
        }
        let hadamard = <<Cpu as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            data_type,
            HadamardTransformOrder::Output,
        )?;
        Ok(Self {
            data_type,
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
        let data_type = self.data_type;

        // Shared A / D / bias pointers.
        let a_buffer_range = a.as_buffer_range_ref();
        let a_byte_off = a_buffer_range.range().start + a_offset * data_type.size_in_bytes();
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
                let b_byte_off = b_buffer_range.range().start + b_offset * data_type.size_in_bytes();
                let b_ptr = SendPtr(
                    unsafe { &*b_buffer_range.buffer().downcast().get() }.as_ptr().wrapping_byte_add(b_byte_off),
                );

                let command_buffer = encoder.as_command_buffer_mut();
                command_buffer.push_command(move || {
                    macro_rules! run {
                        ($ty:ty) => {
                            reference_matmul::<$ty>(
                                a_ptr.as_ptr() as *const $ty,
                                ReferenceWeights::FullPrecision {
                                    weights: b_ptr.as_ptr() as *const $ty,
                                    leading_dimension: leading_dimension_b,
                                    transpose: b_transpose,
                                },
                                d_ptr.as_ptr() as *mut $ty,
                                m_u,
                                n_u,
                                k_u,
                                leading_dimension_a,
                                leading_dimension_d,
                                output_scale,
                                accumulate,
                                bias_ptr.map(|p| p.as_ptr() as *const $ty),
                            )
                        };
                    }
                    match data_type {
                        DataType::F32 => run!(f32),
                        DataType::F16 => run!(f16),
                        DataType::BF16 => run!(bf16),
                        _ => unreachable!(),
                    }
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
                let second_buffer_range = second.as_buffer_range_ref();
                let second_ptr = SendPtr(
                    unsafe { &*second_buffer_range.buffer().get() }
                        .as_ptr()
                        .wrapping_byte_add(second_buffer_range.range().start),
                );

                let command_buffer = encoder.as_command_buffer_mut();
                command_buffer.push_command(move || {
                    macro_rules! run {
                        ($ty:ty) => {
                            reference_matmul::<$ty>(
                                a_ptr.as_ptr() as *const $ty,
                                ReferenceWeights::Quantized(ReferenceQuantWeights {
                                    weights: weights_ptr.as_ptr() as *const u32,
                                    scales: scales_ptr.as_ptr() as *const $ty,
                                    zero_points: use_zero_point.then(|| second_ptr.as_ptr()),
                                    biases: (!use_zero_point).then(|| second_ptr.as_ptr() as *const $ty),
                                    bits,
                                    group_size,
                                }),
                                d_ptr.as_ptr() as *mut $ty,
                                m_u,
                                n_u,
                                k_u,
                                leading_dimension_a,
                                leading_dimension_d,
                                output_scale,
                                accumulate,
                                bias_ptr.map(|p| p.as_ptr() as *const $ty),
                            )
                        };
                    }
                    match data_type {
                        DataType::F32 => run!(f32),
                        DataType::F16 => run!(f16),
                        DataType::BF16 => run!(bf16),
                        _ => unreachable!(),
                    }
                });
            },
        }

        if let Some(factors) = post_rht {
            self.hadamard.encode(d, factors, n, m, encoder);
        }

        Ok(())
    }
}

/// Group-quantized weight operand (weights laid out transposed as [N, K]).
struct ReferenceQuantWeights<T> {
    weights: *const u32,
    scales: *const T,
    /// Exactly one of `zero_points` / `biases` is set, selecting the
    /// scale+zero-point vs scale+bias dequantization.
    zero_points: Option<*const u8>,
    biases: Option<*const T>,
    bits: usize,
    group_size: usize,
}

/// Weight operand for the reference matmul: either full precision or quantized.
enum ReferenceWeights<T> {
    FullPrecision {
        weights: *const T,
        /// Leading dimension (row/col stride) of the weight matrix.
        leading_dimension: usize,
        /// `true` when weights are [N, K] (the GEMV / quantized layout),
        /// `false` when they are [K, N].
        transpose: bool,
    },
    Quantized(ReferenceQuantWeights<T>),
}

/// Single fully-featured CPU reference matmul, used purely for unit-test
/// correctness (performance is irrelevant). For every (i, j) it computes:
///
///   D[i, j] = output_scale * sum_l A[i, l] * Bval[j, l]
///             + (accumulate ? D[i, j] : 0)
///             + (bias       ? bias[j]  : 0)
///
/// where `Bval` is a full-precision weight or a dequantized group-quantized
/// weight. Output transforms beyond scale/accumulate/bias (e.g. RHT) are
/// applied by the caller as a separate pass.
#[allow(clippy::too_many_arguments)]
fn reference_matmul<T: ArrayElement + Float>(
    a: *const T,
    weights: ReferenceWeights<T>,
    output: *mut T,
    m: usize,
    n: usize,
    k: usize,
    leading_dimension_a: usize,
    leading_dimension_d: usize,
    output_scale: f32,
    accumulate: bool,
    bias: Option<*const T>,
) {
    let quant_layout = match &weights {
        ReferenceWeights::Quantized(q) => {
            let num_groups_k = k.div_ceil(q.group_size);
            let zero_point_stride =
                if q.bits == 4 { num_groups_k.div_ceil(2) } else { num_groups_k };
            let pack_factor = if q.bits == 4 { 8 } else { 4 };
            Some((num_groups_k, zero_point_stride, pack_factor))
        },
        ReferenceWeights::FullPrecision { .. } => None,
    };

    unsafe {
        for row in 0..m {
            for col in 0..n {
                let mut accumulator = 0.0f32;
                for inner in 0..k {
                    let a_value = (*a.add(row * leading_dimension_a + inner)).to_f32().unwrap();
                    let b_value = match &weights {
                        ReferenceWeights::FullPrecision {
                            weights,
                            leading_dimension,
                            transpose,
                        } => {
                            let index = if *transpose {
                                col * leading_dimension + inner
                            } else {
                                inner * leading_dimension + col
                            };
                            (*weights.add(index)).to_f32().unwrap()
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
                            let scale = (*q.scales.add(col * num_groups_k + group_index)).to_f32().unwrap();
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
                            } else {
                                (*q.biases.unwrap().add(col * num_groups_k + group_index)).to_f32().unwrap()
                            };
                            scale * quantized_value + bias_term
                        },
                    };
                    accumulator += a_value * b_value;
                }

                let output_index = row * leading_dimension_d + col;
                let mut value = output_scale * accumulator;
                if accumulate {
                    value += (*output.add(output_index)).to_f32().unwrap();
                }
                if let Some(bias) = bias {
                    value += (*bias.add(col)).to_f32().unwrap();
                }
                *output.add(output_index) = T::from(value).unwrap();
            }
        }
    }
}
