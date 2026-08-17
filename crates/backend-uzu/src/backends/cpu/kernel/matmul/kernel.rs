use std::mem::size_of;

use super::reference::{WeightData, read_f32, write_f32};
use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, BufferArg, Encoder, Kernels,
            gpu_types::QuantizationMode,
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{ExpertInput, MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulKernel},
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
    output_rht: ActivationTransform<Cpu>,
    bias_add: <<Cpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel,
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
        })
    }

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Cpu>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Cpu, TB>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        if arguments.gather_indices.is_some() && arguments.expert_routes.is_some() {
            return Err(MatmulError::UnsupportedRouting {
                path: "CpuMatmul",
                reason: "sparse readout and expert routing cannot be combined",
            }
            .into());
        }
        if let Some(routes) = arguments.expert_routes {
            if routes.expert_ids.size() < arguments.m as usize * size_of::<i32>() {
                return Err(MatmulError::UnsupportedRouting {
                    path: "CpuMatmul",
                    reason: "expert_ids must contain at least M entries",
                }
                .into());
            }
            if routes.input == ExpertInput::Tokens && !arguments.m.is_multiple_of(routes.routes_per_token.get()) {
                return Err(MatmulError::UnsupportedRouting {
                    path: "CpuMatmul",
                    reason: "M must be divisible by routes_per_token for token inputs",
                }
                .into());
            }
            if routes.expert_biases.is_some_and(|biases| {
                biases.size()
                    < routes.expert_count.get() as usize * arguments.n as usize * self.weights_data_type.size_in_bytes()
            }) {
                return Err(MatmulError::UnsupportedRouting {
                    path: "CpuMatmul",
                    reason: "expert bias bank must contain expert_count * N values",
                }
                .into());
            }
        }

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
            expert_routes,
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
        #[derive(Clone, Copy)]
        struct ExpertRouteData {
            ids: SendPtr<i32>,
            routes_per_token: usize,
            expert_count: usize,
            input: ExpertInput,
            biases: Option<SendPtr<u8>>,
        }
        let expert_route_data = expert_routes.map(|routes| {
            let ids = routes.expert_ids.as_buffer_range_ref();
            let biases = routes.expert_biases.map(|biases| {
                let range = biases.as_buffer_range_ref();
                SendPtr(unsafe { &*range.buffer().get() }.as_ptr().wrapping_byte_add(range.range().start))
            });
            ExpertRouteData {
                ids: SendPtr(
                    unsafe { &*ids.buffer().get() }.as_ptr().wrapping_byte_add(ids.range().start) as *const i32
                ),
                routes_per_token: routes.routes_per_token.get() as usize,
                expert_count: routes.expert_count.get() as usize,
                input: routes.input,
                biases,
            }
        });
        let d_buffer_range = d.as_buffer_range_mut();
        let d_ptr = SendPtrMut(unsafe {
            (&*d_buffer_range.buffer().get()).as_ptr().wrapping_byte_add(d_buffer_range.range().start) as *mut u8
        });

        let weight_data = WeightData::from_b(b, b_leading_dimension, b_transpose, k_u, n_u);

        let bias_after_rht = post_rht.is_some();
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
                    let (a_row, matrix, valid_route) = match expert_route_data {
                        Some(routes) => {
                            let expert = *routes.ids.as_ptr().add(row);
                            let valid = expert >= 0 && (expert as usize) < routes.expert_count;
                            let a_row = match routes.input {
                                ExpertInput::Tokens => row / routes.routes_per_token,
                                ExpertInput::Routes => row,
                            };
                            (a_row, expert.max(0) as usize, valid)
                        },
                        None => (row, 0, true),
                    };
                    for col in 0..n_u {
                        let output_index = row * n_u + col;
                        if !valid_route {
                            write_f32(d_ptr.as_ptr(), output_data_type, output_index, 0.0);
                            continue;
                        }
                        // Gather remaps output column `col` to B-row `gather_indices[row * n + col]`.
                        let b_col = match gather_ptr {
                            Some(g) => *g.as_ptr().add(row * n_u + col) as usize,
                            None => col,
                        };
                        let mut accumulator = 0.0f32;
                        for inner in 0..k_u {
                            let a_value = match a_data {
                                AData::FullPrecision(ptr) => {
                                    read_f32(ptr.as_ptr(), input_data_type, a_row * k_u + inner)
                                },
                                AData::Int8 {
                                    values,
                                    scales,
                                    group_size,
                                } => {
                                    let groups = k_u.div_ceil(group_size);
                                    let group = inner / group_size;
                                    let q = *(values.as_ptr() as *const i8).add(a_row * k_u + inner) as f32;
                                    let scale = *(scales.as_ptr() as *const f32).add(a_row * groups + group);
                                    q * scale
                                },
                            };
                            let b_value = match &weight_data {
                                WeightData::FullPrecision {
                                    ptr,
                                    leading_dimension,
                                    transpose,
                                } => {
                                    let matrix_stride = if *transpose {
                                        n_u * leading_dimension
                                    } else {
                                        k_u * leading_dimension
                                    };
                                    let index = if *transpose {
                                        matrix * matrix_stride + b_col * leading_dimension + inner
                                    } else {
                                        matrix * matrix_stride + inner * leading_dimension + b_col
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
                                    signed_codes,
                                } => {
                                    let (num_groups_k, zero_point_stride, pack_factor) = quant_layout.unwrap();
                                    let matrix_row = matrix * n_u + b_col;
                                    let weight_linear_index = matrix_row * k_u + inner;
                                    let word_index = weight_linear_index / pack_factor;
                                    let bit_offset = (weight_linear_index % pack_factor) * *bits;
                                    let weights_words = weights.as_ptr() as *const u32;
                                    let word = weights_words.add(word_index).read_unaligned();
                                    let code_mask = (1u32 << bits) - 1;
                                    let mut weight_code = ((word >> bit_offset) & code_mask) as u8;
                                    if *signed_codes {
                                        weight_code ^= 1u8 << (bits - 1);
                                    }
                                    let quantized_value = f32::from(weight_code);
                                    let group_index = inner / group_size;
                                    let scale = read_f32(
                                        scales.as_ptr(),
                                        weights_data_type,
                                        matrix_row * num_groups_k + group_index,
                                    );
                                    let midpoint = (1u32 << (bits - 1)) as f32;
                                    let zero_point = zero_points.map(|zp| {
                                        if *bits == 4 {
                                            let byte_index = matrix_row * zero_point_stride + (group_index >> 1);
                                            let byte_value = *zp.as_ptr().add(byte_index);
                                            if (group_index & 1) == 0 {
                                                (byte_value & 0x0F) as f32
                                            } else {
                                                ((byte_value >> 4) & 0x0F) as f32
                                            }
                                        } else {
                                            *zp.as_ptr().add(matrix_row * zero_point_stride + group_index) as f32
                                        }
                                    });
                                    let bias_term = if let Some(zp) = zero_point {
                                        -scale * zp
                                    } else if let Some(b) = biases {
                                        read_f32(b.as_ptr(), weights_data_type, matrix_row * num_groups_k + group_index)
                                    } else {
                                        -scale * midpoint
                                    };
                                    scale * quantized_value + bias_term
                                },
                            };
                            accumulator += a_value * b_value;
                        }

                        let mut value = output_scale * accumulator;
                        if accumulate {
                            value += read_f32(d_ptr.as_ptr(), output_data_type, output_index);
                        }
                        if !bias_after_rht && let Some(bias) = bias_ptr {
                            value += read_f32(bias.as_ptr(), weights_data_type, col);
                        }
                        if let Some(biases) = expert_route_data.and_then(|routes| routes.biases) {
                            value += read_f32(biases.as_ptr(), weights_data_type, matrix * n_u + col);
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
            self.output_rht.encode_fp_in_place(&mut *d, factors, m, n, encoder);
            if let Some(bias) = bias_alloc {
                let output_length = m.checked_mul(n).expect("matmul output length must fit in u32");
                self.bias_add.encode(None::<&Allocation<Cpu>>, bias, &mut *d, n, output_length, encoder);
            }
        }

        Ok(())
    }
}
