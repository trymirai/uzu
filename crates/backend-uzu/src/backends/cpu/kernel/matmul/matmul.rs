use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
            gpu_types::{HadamardTransformOrder, QuantizationMethod, QuantizationMode, gemm::GemmDTransform},
            kernel::{
                HadamardTransformKernel, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel},
            },
        },
        cpu::{
            BufferDowncastExt, Cpu, context::CpuContext, error::CpuError, kernel::matmul::quant::encode_quantized_gemm,
        },
    },
    dispatch_dtype,
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct MatmulCpuKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    hadamard: <<Cpu as Backend>::Kernels as Kernels>::HadamardTransformKernel,
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
        let hadamard = <<Cpu as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            output_data_type,
            HadamardTransformOrder::Output,
        )?;
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
            hadamard,
            bias_add,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Cpu>>>(
        &mut self,
        arguments: MatmulArguments<Cpu, TB>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        match arguments.b {
            MatmulB::FullPrecision {
                ..
            } => Ok(self.encode_fp(arguments, encoder)?),
            MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            } => Ok(self.encode_quant(arguments, encoder)?),
        }
    }
}

impl MatmulCpuKernel {
    fn encode_fp<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Cpu>>>(
        &mut self,
        arguments: MatmulArguments<Cpu, TB>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), MatmulError<Cpu>> {
        if arguments.d_transform.mask().contains(GemmDTransform::RHT) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::RHT,
                path: "MatmulCpuKernel",
            });
        }

        let ab_scale = arguments.d_transform.ab_scale;
        let bias_alloc = arguments.d_transform.bias;
        let is_accumulate = arguments.d_transform.accumulate;

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
        let weights = match b {
            MatmulB::FullPrecision {
                b: w,
            } => w,
            _ => unreachable!(),
        };

        let command_buffer = encoder.as_command_buffer_mut();
        let m_u = m as usize;
        let n_u = n as usize;
        let k_u = k as usize;
        let lda = k_u;
        let ldb = b_leading_dimension.map(|n| n as usize).unwrap_or(if b_transpose {
            k_u
        } else {
            n_u
        });
        let ldd = n_u;
        let weights_data_type = self.weights_data_type;
        let input_data_type = self.input_data_type;
        let output_data_type = self.output_data_type;
        let a_buffer_range = a.as_buffer_range_ref();
        let b_buffer_range = weights.as_buffer_range_ref();
        let d_buffer_range = d.as_buffer_range_mut();
        let a_byte_off = a_buffer_range.range().start + a_offset;
        let b_byte_off = b_buffer_range.range().start + b_offset;
        let d_byte_off = d_buffer_range.range().start;

        let a_ptr = SendPtr(unsafe { &*a_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(a_byte_off));
        let b_ptr =
            SendPtr(unsafe { &*b_buffer_range.buffer().downcast().get() }.as_ptr().wrapping_byte_add(b_byte_off));
        let d_ptr =
            SendPtrMut(unsafe { (&*d_buffer_range.buffer().get()).as_ptr().wrapping_byte_add(d_byte_off) as *mut u8 });

        let bias_ptr = bias_alloc.map(|bias| {
            let bias_buffer_range = bias.as_buffer_range_ref();
            let bias_range = bias_buffer_range.range();
            SendPtr(unsafe { &*bias_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(bias_range.start))
        });

        command_buffer.push_command(move || {
            dispatch_dtype!(|(TW: weights_data_type, TA: input_data_type, TD: output_data_type)| {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<TA, TW, TD>(
                    a_ptr.as_ptr() as *const TA,
                    b_ptr.as_ptr() as *const TW,
                    ab_scale,
                    d_ptr.as_ptr() as *mut TD,
                    m_u,
                    n_u,
                    k_u,
                    lda,
                    ldb,
                    ldd,
                    b_transpose,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    assert_eq!(
                        weights_data_type, output_data_type,
                        "mixed precision matmul with bias is not supported until bias dtype ownership is settled",
                    );
                    apply_bias::<TD>(d_ptr.as_ptr() as *mut TD, bias.as_ptr() as *const TD, m_u, n_u);
                }
            })
        });

        Ok(())
    }

    fn encode_quant<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Cpu>>>(
        &mut self,
        arguments: MatmulArguments<Cpu, TB>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), MatmulError<Cpu>> {
        let d_mask = arguments.d_transform.mask();
        if d_mask.contains(GemmDTransform::SCALE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::SCALE,
                path: "MatmulCpuKernel/Quant",
            });
        }
        if d_mask.contains(GemmDTransform::ACCUMULATE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: "MatmulCpuKernel/Quant",
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
            return Err(MatmulError::UnsupportedLayout {
                path: "MatmulCpuKernel/Quant",
            });
        }

        let group_size = match arguments.b {
            MatmulB::ScaleBiasDequant {
                group_size,
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                group_size,
                ..
            } => group_size,
            MatmulB::FullPrecision {
                ..
            } => unreachable!(),
        };
        if !matches!(group_size, 16 | 32 | 64 | 128) {
            return Err(MatmulError::UnsupportedGroupSize(group_size as usize));
        }

        let post_rht = arguments.d_transform.rht_factors;
        let post_bias = arguments.d_transform.bias;

        if arguments.m >= 5 && arguments.n > 1 {
            assert_eq!(
                self.input_data_type, self.weights_data_type,
                "mixed precision quantized GEMM input dtype is not supported yet",
            );
            assert_eq!(
                self.output_data_type, self.weights_data_type,
                "mixed precision quantized GEMM output dtype is not supported yet",
            );
            assert!(
                !(post_bias.is_some() && post_rht.is_some()),
                "MatmulCpuKernel/Quant GEMM with both output bias and output RHT is not supported: bias must be applied after RHT",
            );
            let MatmulArguments {
                a,
                a_offset,
                b,
                b_offset,
                b_leading_dimension,
                b_transpose,
                d,
                d_transform,
                m,
                n,
                k,
            } = arguments;
            let d_transform = d_transform.without(GemmDTransform::RHT | GemmDTransform::BIAS);
            encode_quantized_gemm(
                encoder,
                MatmulArguments {
                    a,
                    a_offset,
                    b,
                    b_offset,
                    b_leading_dimension,
                    b_transpose,
                    d: &mut *d,
                    d_transform,
                    m,
                    n,
                    k,
                },
                self.weights_data_type,
            );
            if let Some(bias) = post_bias {
                self.bias_add.encode(None::<&Allocation<Cpu>>, bias, &mut *d, n, m * n, encoder);
            }
            if let Some(factors) = post_rht {
                self.hadamard.encode(d, factors, n, m, encoder);
            }
            return Ok(());
        }

        let hadamard_factors = post_rht;

        let MatmulArguments {
            a,
            a_offset,
            b,
            d,
            m,
            n,
            k,
            ..
        } = arguments;
        let (weights, scales, zp_or_bias, method, mode, group_size) = match b {
            MatmulB::ScaleBiasDequant {
                b: w,
                scales,
                biases,
                mode,
                group_size,
            } => (w, scales, biases, QuantizationMethod::ScaleBias, mode, group_size),
            MatmulB::ScaleZeroPointDequant {
                b: w,
                scales,
                zero_points,
                mode,
                group_size,
            } => (w, scales, zero_points, QuantizationMethod::ScaleZeroPoint, mode, group_size),
            MatmulB::FullPrecision {
                ..
            } => unreachable!(),
        };

        let bits = match mode {
            QuantizationMode::U4 => 4u32,
            QuantizationMode::I8 | QuantizationMode::U8 => 8u32,
        };
        let use_fast = n % 8 == 0 && k % 512 == 0;
        let (zero_points, biases) = match method {
            QuantizationMethod::ScaleZeroPoint => (Some(zp_or_bias), None),
            QuantizationMethod::ScaleBias => (None, Some(zp_or_bias)),
        };

        let context = encoder.context();
        if use_fast {
            let kernel =
                <<Cpu as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                    context,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                    group_size,
                    bits,
                    method,
                    hadamard_factors.is_some(),
                )
                .map_err(MatmulError::BackendError)?;
            kernel.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                &mut *d,
                hadamard_factors,
                k,
                n,
                m,
                encoder,
            );
        } else {
            if hadamard_factors.is_some() {
                return Err(MatmulError::UnsupportedDOp {
                    bit: GemmDTransform::RHT,
                    path: "MatmulCpuKernel/Quant",
                });
            }
            let kernel =
                <<Cpu as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                    context,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                    group_size,
                    bits,
                    method,
                )
                .map_err(MatmulError::BackendError)?;
            kernel.encode(weights, scales, zero_points, biases, (a, a_offset), &mut *d, k, n, m, encoder);
        }
        if let Some(bias) = post_bias {
            self.bias_add.encode(None::<&Allocation<Cpu>>, bias, d, n, m * n, encoder);
        }
        Ok(())
    }
}

fn apply_bias<T: num_traits::Float + Copy>(
    d: *mut T,
    bias: *const T,
    m: usize,
    n: usize,
) {
    unsafe {
        for row in 0..m {
            for col in 0..n {
                let idx = row * n + col;
                *d.add(idx) = *d.add(idx) + *bias.add(col);
            }
        }
    }
}
