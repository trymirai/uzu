use crate::{
    backends::{
        common::{
            Backend, BufferCpuAccessible, BufferMut, BufferRef, Kernels,
            gpu_types::quantization::QuantizationMode,
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{Int8CodeLayout, MatmulA, MatmulArguments, MatmulKernel},
            },
        },
        cpu::{
            Cpu,
            buffer::CpuBufferExt,
            command_buffer::CpuCommandBufferEncoding,
            context::CpuContext,
            kernel::matmul::{
                MatmulCpuError,
                reference::{WeightData, read_bf16, write_bf16},
            },
        },
    },
    data_type::DataType,
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct MatmulCpuKernel {
    output_rht: ActivationTransform<Cpu>,
    bias_add: <<Cpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel,
}

impl MatmulKernel for MatmulCpuKernel {
    type Backend = Cpu;
    type Error = MatmulCpuError;
    type WeightsLayout = ();

    fn new(
        context: &CpuContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Self::Error> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if data_type != DataType::BF16 {
                return Err(MatmulCpuError::UnsupportedDataType(data_type));
            }
        }
        let output_rht = ActivationTransform::output_rht(context, DataType::BF16, None, true)?;
        let bias_add = <<Cpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(
            context,
            DataType::BF16,
            DataType::BF16,
            true,
        )?;
        Ok(Self {
            output_rht,
            bias_add,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Cpu,
            impl BufferRef<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
            impl BufferMut<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
        >,
        command_buffer: &mut CpuCommandBufferEncoding,
    ) -> Result<(), Self::Error> {
        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            output,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;
        let (mut d, ops) = output.into_contiguous(n, "CpuMatmul")?;
        let output_scale = ops.ab_scale;
        let bias_buffer = ops.bias;
        let post_rht = ops.rht_factors;
        let soft_cap = ops.soft_cap;

        let m_u = m as usize;
        let n_u = n as usize;
        let k_u = k as usize;

        #[derive(Clone, Copy)]
        enum AData {
            FullPrecision(SendPtr<u8>),
            Int8 {
                values: SendPtr<u8>,
                scales: SendPtr<u8>,
                group_size: usize,
                code_layout: Int8CodeLayout,
            },
        }
        let a_data = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => {
                let (values, range) = values.parts();
                let byte_offset = range.start + offset * DataType::BF16.size_in_bytes();
                AData::FullPrecision(SendPtr(
                    values.cpu_address().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(byte_offset),
                ))
            },
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums: _,
                scale_group_size: a_group_size,
                code_layout,
            } => {
                let compatible = matches!(a_group_size, 32 | 64 | 128)
                    && k.is_multiple_of(a_group_size)
                    && matches!(b.group_size(), Some(32 | 64 | 128))
                    && b.quantized()
                        .is_some_and(|quantized| matches!(quantized.mode, QuantizationMode::U4 | QuantizationMode::U8));
                if !compatible {
                    return Err(MatmulCpuError::IncompatibleA {
                        path: "CpuMatmul",
                        reason: "symmetric int8 activations require a supported 32/64/128 activation and weight group",
                    });
                }
                let (values, values_range) = values.parts();
                let (scales, scales_range) = scales.parts();
                AData::Int8 {
                    values: SendPtr(
                        values.cpu_address().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(values_range.start),
                    ),
                    scales: SendPtr(
                        scales.cpu_address().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(scales_range.start),
                    ),
                    group_size: a_group_size as usize,
                    code_layout,
                }
            },
            MatmulA::Trellis {
                ..
            } => {
                return Err(MatmulCpuError::UnsupportedLayout {
                    path: "CpuMatmul",
                });
            },
        };
        let bias_ptr = bias_buffer.map(|bias| SendPtr(bias.cpu_ptr().as_ptr().cast::<u8>().cast_const()));
        let gather_ptr = gather_indices.map(|indices| {
            let (indices, range) = indices.parts();
            SendPtr(indices.cpu_address().as_ptr().cast::<u32>().cast_const().wrapping_byte_add(range.start))
        });
        let (d_buffer, d_range) = d.reborrow().parts();
        let d_ptr = SendPtrMut(d_buffer.cpu_address().as_ptr().cast::<u8>().wrapping_byte_add(d_range.start));

        let weight_data = WeightData::from_b(b, b_leading_dimension, b_transpose, k_u, n_u)?;

        let bias_after_rht = post_rht.is_some();
        command_buffer.push_command(move || {
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
                            let a_value = match a_data {
                                AData::FullPrecision(ptr) => read_bf16(ptr.as_ptr(), row * k_u + inner),
                                AData::Int8 {
                                    values,
                                    scales,
                                    group_size,
                                    code_layout,
                                } => {
                                    let groups = k_u.div_ceil(group_size);
                                    let group = inner / group_size;
                                    let code_index = code_layout.index(inner);
                                    let q = *(values.as_ptr() as *const i8).add(row * k_u + code_index) as f32;
                                    let scale = *(scales.as_ptr() as *const f32).add(row * groups + group);
                                    q * scale
                                },
                            };
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
                                    read_bf16(ptr.as_ptr(), index)
                                },
                                WeightData::Quantized {
                                    weights,
                                    scales,
                                    zero_points,
                                    biases,
                                    scale_strides,
                                    bits,
                                    group_size,
                                    signed_codes,
                                } => {
                                    let pack_factor = 32 / *bits;
                                    let weight_linear_index = b_col * k_u + inner;
                                    let word_index = weight_linear_index / pack_factor;
                                    let code_index_in_word = weight_linear_index % pack_factor;
                                    let bit_offset = code_index_in_word * *bits;
                                    let weights_words = weights.as_ptr() as *const u32;
                                    let word = weights_words.add(word_index).read_unaligned();
                                    let code_mask = (1u32 << bits) - 1;
                                    let mut weight_code = ((word >> bit_offset) & code_mask) as u8;
                                    if *signed_codes {
                                        weight_code ^= 1u8 << (bits - 1);
                                    }
                                    let quantized_value = f32::from(weight_code);
                                    let group_index = inner / group_size;
                                    let metadata_index = b_col * scale_strides.output_stride as usize
                                        + group_index * scale_strides.group_stride as usize;
                                    let scale = read_bf16(scales.as_ptr(), metadata_index);
                                    let midpoint = (1u32 << (bits - 1)) as f32;
                                    let zero_point = zero_points.as_ref().map(|(zp, strides)| {
                                        let zero_point_index = b_col * strides.output_stride as usize
                                            + group_index * strides.group_stride as usize;
                                        if *bits == 4 {
                                            let byte_index = zero_point_index / 2;
                                            let byte_value = *zp.as_ptr().add(byte_index);
                                            if zero_point_index.is_multiple_of(2) {
                                                (byte_value & 0x0F) as f32
                                            } else {
                                                ((byte_value >> 4) & 0x0F) as f32
                                            }
                                        } else {
                                            *zp.as_ptr().add(zero_point_index) as f32
                                        }
                                    });
                                    let bias_term = if let Some(zp) = zero_point {
                                        -scale * zp
                                    } else if let Some(b) = biases {
                                        read_bf16(b.as_ptr(), metadata_index)
                                    } else {
                                        -scale * midpoint
                                    };
                                    scale * quantized_value + bias_term
                                },
                            };
                            accumulator += a_value * b_value;
                        }

                        let output_index = row * n_u + col;
                        let mut value = output_scale * accumulator;
                        if !bias_after_rht && let Some(bias) = bias_ptr {
                            value += read_bf16(bias.as_ptr(), col);
                        }
                        if let Some(cap) = soft_cap {
                            value = cap * (value / cap).tanh();
                        }
                        write_bf16(d_ptr.as_ptr(), output_index, value);
                    }
                }
            }
        });

        if let Some(factors) = post_rht {
            self.output_rht.encode_fp_in_place(d.reborrow(), factors, None, m, n, command_buffer);
            if let Some(bias) = bias_buffer {
                let output_length = m.checked_mul(n).expect("matmul output length must fit in u32");
                self.bias_add.encode(
                    None::<&<Cpu as Backend>::ScratchBuffer>,
                    bias,
                    d,
                    n,
                    output_length,
                    command_buffer,
                );
            }
        }

        Ok(())
    }
}
