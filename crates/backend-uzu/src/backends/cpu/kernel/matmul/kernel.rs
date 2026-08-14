use half::{bf16, f16};

use super::reference::{WeightData, read_f32, write_f32};
use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, BufferArg, Encoder, Kernels,
            gpu_types::QuantizationMode,
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulKernel},
            },
        },
        cpu::{Cpu, context::CpuContext, error::CpuError, parallel::Pool},
    },
    data_type::DataType,
    utils::pointers::{SendPtr, SendPtrMut},
};

const LANES: usize = 4;

#[derive(Clone, Copy)]
enum AData {
    FullPrecision(SendPtr<u8>),
    Int8 {
        values: SendPtr<u8>,
        scales: SendPtr<u8>,
        group_size: usize,
    },
}

#[inline(always)]
unsafe fn decode_typed<W>(
    out: &mut Vec<f32>,
    base: *const W,
    offset: usize,
    k: usize,
) where
    W: Copy,
    f32: From<W>,
{
    unsafe {
        let source = base.add(offset);
        out.extend((0..k).map(|index| f32::from(*source.add(index))));
    }
}

unsafe fn decode_activation_row(
    out: &mut Vec<f32>,
    a: AData,
    input_data_type: DataType,
    row: usize,
    k: usize,
) {
    unsafe {
        match a {
            AData::FullPrecision(values) => {
                let base = values.as_ptr();
                match input_data_type {
                    DataType::F32 => decode_typed(out, base as *const f32, row * k, k),
                    DataType::F16 => decode_typed(out, base as *const f16, row * k, k),
                    DataType::BF16 => decode_typed(out, base as *const bf16, row * k, k),
                    _ => unreachable!(),
                }
            },
            AData::Int8 {
                values,
                scales,
                group_size,
            } => {
                let groups = k.div_ceil(group_size);
                let codes = (values.as_ptr() as *const i8).add(row * k);
                let scales = (scales.as_ptr() as *const f32).add(row * groups);
                out.extend((0..k).map(|index| *codes.add(index) as f32 * *scales.add(index / group_size)));
            },
        }
    }
}

#[inline(always)]
unsafe fn dot_full_precision_typed<W>(
    a_row: &[f32],
    base: *const W,
    leading_dimension: usize,
    transpose: bool,
    b_col: usize,
) -> f32
where
    W: Copy,
    f32: From<W>,
{
    let mut accumulator = 0.0f32;
    unsafe {
        if transpose {
            let column = base.add(b_col * leading_dimension);
            let (blocks, remainder) = a_row.as_chunks::<LANES>();
            let mut chains = [0.0f32; LANES];
            let mut inner = 0usize;
            for block in blocks {
                for chain in 0..LANES {
                    chains[chain] += block[chain] * f32::from(*column.add(inner + chain));
                }
                inner += LANES;
            }
            for activation in remainder {
                accumulator += activation * f32::from(*column.add(inner));
                inner += 1;
            }
            accumulator += (chains[0] + chains[1]) + (chains[2] + chains[3]);
        } else {
            for (inner, activation) in a_row.iter().enumerate() {
                accumulator += activation * f32::from(*base.add(inner * leading_dimension + b_col));
            }
        }
    }
    accumulator
}

#[inline]
unsafe fn dot_full_precision(
    a_row: &[f32],
    weights: *const u8,
    weights_data_type: DataType,
    leading_dimension: usize,
    transpose: bool,
    b_col: usize,
) -> f32 {
    unsafe {
        match weights_data_type {
            DataType::F32 => {
                dot_full_precision_typed(a_row, weights as *const f32, leading_dimension, transpose, b_col)
            },
            DataType::F16 => {
                dot_full_precision_typed(a_row, weights as *const f16, leading_dimension, transpose, b_col)
            },
            DataType::BF16 => {
                dot_full_precision_typed(a_row, weights as *const bf16, leading_dimension, transpose, b_col)
            },
            _ => unreachable!(),
        }
    }
}

#[inline]
unsafe fn dot_quantized(
    a_row: &[f32],
    weight_data: &WeightData,
    layout: (usize, usize, usize),
    weights_data_type: DataType,
    b_col: usize,
) -> f32 {
    let WeightData::Quantized {
        weights,
        scales,
        zero_points,
        biases,
        bits,
        group_size,
        signed_codes,
    } = weight_data
    else {
        unreachable!();
    };
    let (num_groups_k, zero_point_stride, pack_factor) = layout;
    let bits = *bits;
    let group_size = *group_size;
    let code_mask = (1u32 << bits) - 1;
    let midpoint = (1u32 << (bits - 1)) as f32;
    let sign_flip = if *signed_codes {
        1u8 << (bits - 1)
    } else {
        0
    };
    let k = a_row.len();

    let mut accumulator = 0.0f32;
    unsafe {
        let words = weights.as_ptr() as *const u32;
        let column_start = b_col * k;

        let mut inner = 0usize;
        let mut group = 0usize;
        while inner < k {
            let group_end = ((group + 1) * group_size).min(k);
            let scale = read_f32(scales.as_ptr(), weights_data_type, b_col * num_groups_k + group);
            let bias_term = if let Some(zero_points) = zero_points {
                let zero_point = if bits == 4 {
                    let byte = *zero_points.as_ptr().add(b_col * zero_point_stride + (group >> 1));
                    if group & 1 == 0 {
                        (byte & 0x0F) as f32
                    } else {
                        ((byte >> 4) & 0x0F) as f32
                    }
                } else {
                    *zero_points.as_ptr().add(b_col * zero_point_stride + group) as f32
                };
                -scale * zero_point
            } else if let Some(biases) = biases {
                read_f32(biases.as_ptr(), weights_data_type, b_col * num_groups_k + group)
            } else {
                -scale * midpoint
            };

            while inner < group_end {
                let linear = column_start + inner;
                let word = words.add(linear / pack_factor).read_unaligned();
                let mut slot = linear % pack_factor;
                while slot < pack_factor && inner < group_end {
                    let code = (((word >> (slot * bits)) & code_mask) as u8) ^ sign_flip;
                    accumulator += a_row[inner] * (scale * f32::from(code) + bias_term);
                    slot += 1;
                    inner += 1;
                }
            }

            group += 1;
        }
    }
    accumulator
}

pub struct MatmulCpuKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    output_rht: ActivationTransform<Cpu>,
    bias_add: <<Cpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel,
    pool: std::sync::Arc<Pool>,
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
        let output_rht = ActivationTransform::output_rht(context, output_data_type, true)?;
        let bias_add = <<Cpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(
            context,
            output_data_type,
            weights_data_type,
            true,
        )?;
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            output_rht,
            bias_add,
            pool: context.pool().clone(),
        })
    }

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Cpu>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Cpu, TB>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        let output_scale = arguments.d_transform.ab_scale;
        let accumulate = arguments.d_transform.accumulate;
        let bias_alloc = arguments.d_transform.bias;
        let post_rht = arguments.d_transform.rht_factors;
        let soft_cap = arguments.d_transform.soft_cap;

        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;

        let m_u = m as usize;
        let n_u = n as usize;
        let k_u = k as usize;
        let weights_data_type = self.weights_data_type;
        let input_data_type = self.input_data_type;
        let output_data_type = self.output_data_type;

        let a_data = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => {
                let range = values.as_buffer_range_ref();
                let byte_offset = range.range().start + offset * input_data_type.size_in_bytes();
                AData::FullPrecision(SendPtr(unsafe { &*range.buffer().get() }.as_ptr().wrapping_byte_add(byte_offset)))
            },
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums: _,
                group_size: a_group_size,
            } => {
                let compatible = matches!(a_group_size, 32 | 64 | 128)
                    && k.is_multiple_of(a_group_size)
                    && matches!(b.group_size(), Some(32 | 64 | 128))
                    && matches!(
                        b,
                        MatmulB::ScaleSymmetricDequant {
                            mode: QuantizationMode::U4 | QuantizationMode::U8,
                            ..
                        } | MatmulB::ScaleBiasDequant {
                            mode: QuantizationMode::U4 | QuantizationMode::U8,
                            ..
                        } | MatmulB::ScaleZeroPointDequant {
                            mode: QuantizationMode::U4 | QuantizationMode::U8,
                            ..
                        }
                    );
                if !compatible {
                    return Err(MatmulError::IncompatibleA {
                        path: "CpuMatmul",
                        reason: "symmetric int8 activations require a supported 32/64/128 activation and weight group",
                    }
                    .into());
                }
                let values_range = values.as_buffer_range_ref();
                let scales_range = scales.as_buffer_range_ref();
                AData::Int8 {
                    values: SendPtr(
                        unsafe { &*values_range.buffer().get() }.as_ptr().wrapping_byte_add(values_range.range().start),
                    ),
                    scales: SendPtr(
                        unsafe { &*scales_range.buffer().get() }.as_ptr().wrapping_byte_add(scales_range.range().start),
                    ),
                    group_size: a_group_size as usize,
                }
            },
        };
        let bias_ptr = bias_alloc.map(|bias| {
            let r = bias.as_buffer_range_ref();
            SendPtr(unsafe { &*r.buffer().get() }.as_ptr().wrapping_byte_add(r.range().start))
        });
        let gather_ptr = gather_indices.map(|indices| {
            let r = indices.as_buffer_range_ref();
            SendPtr(unsafe { &*r.buffer().get() }.as_ptr().wrapping_byte_add(r.range().start) as *const u32)
        });
        let d_buffer_range = d.as_buffer_range_mut();
        let d_ptr = SendPtrMut(unsafe {
            (&*d_buffer_range.buffer().get()).as_ptr().wrapping_byte_add(d_buffer_range.range().start) as *mut u8
        });

        let weight_data = WeightData::from_b(b, b_leading_dimension, b_transpose, k_u, n_u);

        let bias_after_rht = post_rht.is_some();
        let pool = self.pool.clone();
        let command_buffer = encoder.as_command_buffer_mut();
        command_buffer.push_command(move || {
            let quant_layout = match &weight_data {
                WeightData::Quantized {
                    bits,
                    group_size,
                    ..
                } => {
                    let num_groups_k = k_u.div_ceil(*group_size);
                    let zero_point_stride = if *bits == 4 {
                        num_groups_k.div_ceil(2)
                    } else {
                        num_groups_k
                    };
                    let pack_factor = if *bits == 4 {
                        8
                    } else {
                        4
                    };
                    Some((num_groups_k, zero_point_stride, pack_factor))
                },
                WeightData::FullPrecision {
                    ..
                } => None,
            };

            let mut activations = Vec::with_capacity(m_u * k_u);
            for row in 0..m_u {
                unsafe { decode_activation_row(&mut activations, a_data, input_data_type, row, k_u) };
            }

            let compute_columns = |columns: std::ops::Range<usize>| unsafe {
                for row in 0..m_u {
                    let activation_row = &activations[row * k_u..(row + 1) * k_u];
                    for col in columns.clone() {
                        // Gather remaps output column `col` to B-row `gather_indices[row * n + col]`.
                        let b_col = match gather_ptr {
                            Some(g) => *g.as_ptr().add(row * n_u + col) as usize,
                            None => col,
                        };
                        let accumulator = match &weight_data {
                            WeightData::FullPrecision {
                                ptr,
                                leading_dimension,
                                transpose,
                            } => dot_full_precision(
                                activation_row,
                                ptr.as_ptr(),
                                weights_data_type,
                                *leading_dimension,
                                *transpose,
                                b_col,
                            ),
                            WeightData::Quantized {
                                ..
                            } => dot_quantized(
                                activation_row,
                                &weight_data,
                                quant_layout.unwrap(),
                                weights_data_type,
                                b_col,
                            ),
                        };

                        let output_index = row * n_u + col;
                        let mut value = output_scale * accumulator;
                        if accumulate {
                            value += read_f32(d_ptr.as_ptr(), output_data_type, output_index);
                        }
                        if !bias_after_rht && let Some(bias) = bias_ptr {
                            value += read_f32(bias.as_ptr(), weights_data_type, col);
                        }
                        if let Some(cap) = soft_cap {
                            value = cap * (value / cap).tanh();
                        }
                        write_f32(d_ptr.as_ptr(), output_data_type, output_index, value);
                    }
                }
            };

            pool.for_each_chunk(n_u, m_u * n_u * k_u, compute_columns);
        });

        if let Some(factors) = post_rht {
            self.output_rht.encode_fp_in_place(&mut *d, factors, m, n, encoder);
            if let Some(bias) = bias_alloc {
                let output_length = m.checked_mul(n).expect("matmul output length must fit in u32");
                self.bias_add.encode(None::<&Allocation<Cpu>>, bias, &mut *d, n, output_length, encoder);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/backends/cpu/kernel/matmul/parallel_matmul_test.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/backends/cpu/kernel/matmul/quant_matmul_test.rs"]
mod quant_tests;
