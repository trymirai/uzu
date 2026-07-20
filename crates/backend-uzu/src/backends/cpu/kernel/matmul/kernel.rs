use super::reference::{WeightData, read_f32, write_f32};
use crate::{
    backends::{
        common::{
            AsBufferRangeMut, AsBufferRangeRef, Backend, BufferArg, Encoder, Kernels,
            gpu_types::HadamardTransformOrder,
            kernel::{
                HadamardTransformKernel,
                matmul::{MatmulArguments, MatmulError, MatmulKernel},
            },
        },
        cpu::{Cpu, context::CpuContext, error::CpuError},
    },
    data_type::DataType,
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
            a_offset,
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

        let a_buffer_range = a.as_buffer_range_ref();
        let a_byte_off = a_buffer_range.range().start + a_offset * input_data_type.size_in_bytes();
        let a_ptr = SendPtr(unsafe { &*a_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(a_byte_off));
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

            unsafe {
                for row in 0..m_u {
                    for col in 0..n_u {
                        // Gather remaps output column `col` to B-row `gather_indices[row * n + col]`.
                        let b_col = match gather_ptr {
                            Some(g) => *g.as_ptr().add(row * n_u + col) as usize,
                            None => col,
                        };
                        let mut accumulator = 0.0f32;
                        for inner in 0..k_u {
                            let a_value = read_f32(a_ptr.as_ptr(), input_data_type, row * k_u + inner);
                            let b_value = match &weight_data {
                                WeightData::FullPrecision {
                                    ptr,
                                    leading_dimension,
                                    transpose,
                                } => {
                                    let index = if *transpose {
                                        b_col * leading_dimension + inner
                                    } else {
                                        inner * leading_dimension + b_col
                                    };
                                    read_f32(ptr.as_ptr(), weights_data_type, index)
                                },
                                WeightData::Quantized {
                                    weights,
                                    scales,
                                    zero_points,
                                    biases,
                                    bits,
                                    group_size,
                                } => {
                                    let (num_groups_k, zero_point_stride, pack_factor) = quant_layout.unwrap();
                                    let weight_linear_index = b_col * k_u + inner;
                                    let quantized_value = if *bits == 4 {
                                        let word_index = weight_linear_index / pack_factor;
                                        let bit_offset = (weight_linear_index % pack_factor) * 4;
                                        let w = weights.as_ptr() as *const u32;
                                        ((w.add(word_index).read_unaligned() >> bit_offset) & 0xF) as f32
                                    } else {
                                        let word_index = weight_linear_index / pack_factor;
                                        let bit_offset = (weight_linear_index % pack_factor) * 8;
                                        let w = weights.as_ptr() as *const u32;
                                        ((w.add(word_index).read_unaligned() >> bit_offset) & 0xFF) as f32
                                    };
                                    let group_index = inner / group_size;
                                    let scale = read_f32(
                                        scales.as_ptr(),
                                        weights_data_type,
                                        b_col * num_groups_k + group_index,
                                    );
                                    let bias_term = if let Some(zp) = zero_points {
                                        let zero_point = if *bits == 4 {
                                            let byte_index = b_col * zero_point_stride + (group_index >> 1);
                                            let byte_value = *zp.as_ptr().add(byte_index);
                                            if (group_index & 1) == 0 {
                                                (byte_value & 0x0F) as f32
                                            } else {
                                                ((byte_value >> 4) & 0x0F) as f32
                                            }
                                        } else {
                                            *zp.as_ptr().add(b_col * zero_point_stride + group_index) as f32
                                        };
                                        -scale * zero_point
                                    } else if let Some(b) = biases {
                                        read_f32(b.as_ptr(), weights_data_type, b_col * num_groups_k + group_index)
                                    } else {
                                        let midpoint = (1u32 << (bits - 1)) as f32;
                                        -scale * midpoint
                                    };
                                    scale * quantized_value + bias_term
                                },
                            };
                            accumulator += a_value * b_value;
                        }

                        let output_index = row * n_u + col;
                        let mut value = output_scale * accumulator;
                        if accumulate {
                            value += read_f32(d_ptr.as_ptr(), output_data_type, output_index);
                        }
                        if let Some(bias) = bias_ptr {
                            value += read_f32(bias.as_ptr(), weights_data_type, col);
                        }
                        if let Some(cap) = soft_cap {
                            value = cap * (value / cap).tanh();
                        }
                        write_f32(d_ptr.as_ptr(), output_data_type, output_index, value);
                    }
                }
            }
        });

        if let Some(factors) = post_rht {
            self.hadamard.encode(d, factors, n, m, encoder);
        }

        Ok(())
    }
}
