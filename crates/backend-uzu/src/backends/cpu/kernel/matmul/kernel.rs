use super::reference::{WeightData, read_f32, write_f32};
use crate::{
    backends::{
        common::{
            AsBufferRangeMut, AsBufferRangeRef, Backend, BufferArg, Encoder, Kernels,
            gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder},
            kernel::{
                HadamardTransformKernel,
                matmul::{MatmulA, MatmulArguments, MatmulError, MatmulKernel},
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

        let MatmulArguments {
            a,
            b,
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
        let weights_data_type = self.weights_data_type;
        let input_data_type = self.input_data_type;
        let output_data_type = self.output_data_type;

        #[derive(Clone, Copy)]
        enum AData {
            FullPrecision(SendPtr<u8>),
            Int8 {
                values: SendPtr<u8>,
                scales: SendPtr<u8>,
                zero_points: Option<SendPtr<u8>>,
                group_size: usize,
            },
        }
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
                group_size,
                ..
            } => {
                if group_size == 0
                    || !group_size.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE as u32)
                    || b.group_size() != Some(group_size)
                    || b.bits_per_b() != Some(8)
                {
                    return Err(MatmulError::IncompatibleA {
                        path: "CpuMatmul",
                        reason: "int8 activation groups must be non-zero RHT-block multiples and match 8-bit weights",
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
                    zero_points: None,
                    group_size: group_size as usize,
                }
            },
            MatmulA::Int8Asymmetric {
                values,
                scales,
                zero_points,
                group_size,
                ..
            } => {
                if group_size == 0
                    || !group_size.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE as u32)
                    || b.group_size() != Some(group_size)
                    || b.bits_per_b() != Some(8)
                {
                    return Err(MatmulError::IncompatibleA {
                        path: "CpuMatmul",
                        reason: "int8 activation groups must be non-zero RHT-block multiples and match 8-bit weights",
                    }
                    .into());
                }
                let values_range = values.as_buffer_range_ref();
                let scales_range = scales.as_buffer_range_ref();
                let zp_range = zero_points.as_buffer_range_ref();
                AData::Int8 {
                    values: SendPtr(
                        unsafe { &*values_range.buffer().get() }.as_ptr().wrapping_byte_add(values_range.range().start),
                    ),
                    scales: SendPtr(
                        unsafe { &*scales_range.buffer().get() }.as_ptr().wrapping_byte_add(scales_range.range().start),
                    ),
                    zero_points: Some(SendPtr(
                        unsafe { &*zp_range.buffer().get() }.as_ptr().wrapping_byte_add(zp_range.range().start),
                    )),
                    group_size: group_size as usize,
                }
            },
        };
        let bias_ptr = bias_alloc.map(|bias| {
            let r = bias.as_buffer_range_ref();
            SendPtr(unsafe { &*r.buffer().get() }.as_ptr().wrapping_byte_add(r.range().start))
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
                        let mut accumulator = 0.0f32;
                        for inner in 0..k_u {
                            let a_value = match a_data {
                                AData::FullPrecision(ptr) => read_f32(ptr.as_ptr(), input_data_type, row * k_u + inner),
                                AData::Int8 {
                                    values,
                                    scales,
                                    zero_points,
                                    group_size,
                                } => {
                                    let groups = k_u.div_ceil(group_size);
                                    let group = inner / group_size;
                                    let q = *(values.as_ptr() as *const i8).add(row * k_u + inner) as f32;
                                    let scale = *(scales.as_ptr() as *const f32).add(row * groups + group);
                                    let zp = zero_points
                                        .map(|zp| *(zp.as_ptr() as *const i8).add(row * groups + group) as f32)
                                        .unwrap_or(0.0);
                                    (q - zp) * scale
                                },
                            };
                            let b_value = match &weight_data {
                                WeightData::FullPrecision {
                                    ptr,
                                    leading_dimension,
                                    transpose,
                                } => {
                                    let index = if *transpose {
                                        col * leading_dimension + inner
                                    } else {
                                        inner * leading_dimension + col
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
                                    let weight_linear_index = col * k_u + inner;
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
                                    let scale =
                                        read_f32(scales.as_ptr(), weights_data_type, col * num_groups_k + group_index);
                                    let bias_term = if let Some(zp) = zero_points {
                                        let zero_point = if *bits == 4 {
                                            let byte_index = col * zero_point_stride + (group_index >> 1);
                                            let byte_value = *zp.as_ptr().add(byte_index);
                                            if (group_index & 1) == 0 {
                                                (byte_value & 0x0F) as f32
                                            } else {
                                                ((byte_value >> 4) & 0x0F) as f32
                                            }
                                        } else {
                                            *zp.as_ptr().add(col * zero_point_stride + group_index) as f32
                                        };
                                        -scale * zero_point
                                    } else if let Some(b) = biases {
                                        read_f32(b.as_ptr(), weights_data_type, col * num_groups_k + group_index)
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
