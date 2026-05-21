use half::{bf16, f16};

use crate::{
    DataType,
    backends::{
        common::{
            AsBufferRangeMut, AsBufferRangeRef, Encoder, Kernels,
            gpu_types::{
                QuantizationMethod, QuantizationMode,
                gemm::GemmDTransform,
            },
            kernel::{
                QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel,
                matmul::{
                    MatmulArguments, MatmulB, MatmulError, MatmulKernel, ResolvedDTransform, resolve_a, resolve_d,
                },
            },
        },
        cpu::{Cpu, context::CpuContext, kernel::matmul::quant::encode_quantized_gemm},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct MatmulCpuKernel {
    data_type: DataType,
}

impl MatmulKernel for MatmulCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &CpuContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Cpu>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }
        Ok(Self {
            data_type,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<Cpu>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), MatmulError<Cpu>> {
        let resolved_a = resolve_a(&arguments.a_prologue);
        if !resolved_a.mask.is_empty() {
            return Err(MatmulError::UnsupportedAOp {
                bit: resolved_a.mask,
                path: "MatmulCpuKernel",
            });
        }
        match arguments.b {
            MatmulB::FullPrecision { .. } => self.encode_fp(arguments, encoder),
            MatmulB::ScaleBiasDequant { .. } | MatmulB::ScaleZeroPointDequant { .. } => {
                self.encode_quant(arguments, encoder)
            },
        }
    }
}

impl MatmulCpuKernel {
    fn encode_fp(
        &mut self,
        arguments: MatmulArguments<Cpu>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), MatmulError<Cpu>> {
        let resolved_d = resolve_d(&arguments.d_transform);
        if resolved_d.mask.contains(GemmDTransform::RHT) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::RHT,
                path: "MatmulCpuKernel",
            });
        }

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
        let ldb = b_leading_dimension.map(|n| n as usize).unwrap_or(if b_transpose { k_u } else { n_u });
        let ldd = n_u;
        let data_type = self.data_type;
        let a_buffer_range = a.as_buffer_range_ref();
        let b_buffer_range = weights.as_buffer_range_ref();
        let d_buffer_range = d.as_buffer_range_mut();
        let a_byte_off = a_buffer_range.range().start + a_offset;
        let b_byte_off = b_buffer_range.range().start + b_offset;
        let d_byte_off = d_buffer_range.range().start;

        let a_ptr = SendPtr(unsafe { &*a_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(a_byte_off));
        let b_ptr = SendPtr(unsafe { &*b_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(b_byte_off));
        let d_ptr =
            SendPtrMut(unsafe { (&*d_buffer_range.buffer().get()).as_ptr().wrapping_byte_add(d_byte_off) as *mut u8 });

        let is_accumulate = resolved_d.mask.contains(GemmDTransform::ACCUMULATE);
        let ab_scale = resolved_d.ab_scale;
        let bias_ptr = resolved_d.bias.map(|bias| {
            let bias_buffer_range = bias.as_buffer_range_ref();
            let bias_range = bias_buffer_range.range();
            SendPtr(unsafe { &*bias_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(bias_range.start))
        });

        command_buffer.push_command(move || match data_type {
            DataType::F32 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f32>(
                    a_ptr.as_ptr() as *const f32,
                    b_ptr.as_ptr() as *const f32,
                    ab_scale,
                    d_ptr.as_ptr() as *mut f32,
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
                    apply_bias::<f32>(d_ptr.as_ptr() as *mut f32, bias.as_ptr() as *const f32, m_u, n_u);
                }
            },
            DataType::F16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f16>(
                    a_ptr.as_ptr() as *const f16,
                    b_ptr.as_ptr() as *const f16,
                    ab_scale,
                    d_ptr.as_ptr() as *mut f16,
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
                    apply_bias::<f16>(d_ptr.as_ptr() as *mut f16, bias.as_ptr() as *const f16, m_u, n_u);
                }
            },
            DataType::BF16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<bf16>(
                    a_ptr.as_ptr() as *const bf16,
                    b_ptr.as_ptr() as *const bf16,
                    ab_scale,
                    d_ptr.as_ptr() as *mut bf16,
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
                    apply_bias::<bf16>(d_ptr.as_ptr() as *mut bf16, bias.as_ptr() as *const bf16, m_u, n_u);
                }
            },
            _ => unreachable!(),
        });

        Ok(())
    }

    fn encode_quant(
        &mut self,
        arguments: MatmulArguments<Cpu>,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), MatmulError<Cpu>> {
        let resolved_d = resolve_d(&arguments.d_transform);
        if resolved_d.mask.contains(GemmDTransform::SCALE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::SCALE,
                path: "MatmulCpuKernel/Quant",
            });
        }
        if resolved_d.mask.contains(GemmDTransform::ACCUMULATE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: "MatmulCpuKernel/Quant",
            });
        }
        if resolved_d.mask.contains(GemmDTransform::BIAS) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::BIAS,
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
            MatmulB::FullPrecision { .. } => unreachable!(),
        };
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MatmulError::UnsupportedGroupSize(group_size as usize));
        }

        if arguments.m >= 5 && arguments.n > 1 {
            encode_quantized_gemm(encoder, arguments, resolved_d, self.data_type);
            return Ok(());
        }

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
            MatmulB::FullPrecision { .. } => unreachable!(),
        };

        let bits = match mode {
            QuantizationMode::U4 => 4u32,
            QuantizationMode::I8 | QuantizationMode::U8 => 8u32,
        };
        let use_fast = n % 8 == 0 && k % 512 == 0;
        let hadamard_factors = resolved_d.rht;
        let (zero_points, biases) = match method {
            QuantizationMethod::ScaleZeroPoint => (Some(zp_or_bias), None),
            QuantizationMethod::ScaleBias => (None, Some(zp_or_bias)),
        };

        let context = encoder.context();
        if use_fast {
            let kernel =
                <<Cpu as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                    context,
                    self.data_type,
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
                d,
                hadamard_factors,
                k,
                n,
                m,
                encoder,
            );
        } else {
            if hadamard_factors.is_some() {
                return Err(MatmulError::UnsupportedHadamard);
            }
            let kernel = <<Cpu as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                context,
                self.data_type,
                group_size,
                bits,
                method,
            )
            .map_err(MatmulError::BackendError)?;
            kernel.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                d,
                k,
                n,
                m,
                encoder,
            );
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
