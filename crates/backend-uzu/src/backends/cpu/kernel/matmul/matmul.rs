use half::{bf16, f16};

use crate::{
    DataType,
    backends::{
        common::{
            AsBufferRangeMut, AsBufferRangeRef, Encoder, Kernels,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
                quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError},
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
    ) {
        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            ab_scale,
            c,
            d,
            batch_dim,
            input_dim,
            output_dim,
        } = arguments;
        let command_buffer = encoder.as_command_buffer_mut();
        let m = batch_dim as usize;
        let n = output_dim as usize;
        let k = input_dim as usize;
        let lda = k;
        let ldb = b_leading_dimension.map(|n| n as usize).unwrap_or(if b_transpose {
            k
        } else {
            n
        });
        let ldd = n;
        let data_type = self.data_type;
        let a_buffer_range = a.as_buffer_range_ref();
        let b_buffer_range = b.as_buffer_range_ref();
        let d_buffer_range = d.as_buffer_range_mut();
        let a_offset = a_buffer_range.range().start + a_offset;
        let b_offset = b_buffer_range.range().start + b_offset;
        let d_offset = d_buffer_range.range().start;

        let a_ptr = SendPtr(unsafe { &*a_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(a_offset));
        let b_ptr = SendPtr(unsafe { &*b_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(b_offset));
        let d_ptr =
            SendPtrMut(unsafe { (&*d_buffer_range.buffer().get()).as_ptr().wrapping_byte_add(d_offset) as *mut u8 });

        let (is_accumulate, bias_ptr) = match c {
            MatmulArgumentC::Accumulate => (true, None),
            MatmulArgumentC::Bias(bias) => {
                let bias_buffer_range = bias.as_buffer_range_ref();
                let bias_range = bias_buffer_range.range();
                (
                    false,
                    Some(SendPtr(
                        unsafe { &*bias_buffer_range.buffer().get() }.as_ptr().wrapping_byte_add(bias_range.start),
                    )),
                )
            },
            MatmulArgumentC::None => (false, None),
        };

        command_buffer.push_command(move || match data_type {
            DataType::F32 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f32>(
                    a_ptr.as_ptr() as *const f32,
                    b_ptr.as_ptr() as *const f32,
                    ab_scale,
                    d_ptr.as_ptr() as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                    b_transpose,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<f32>(d_ptr.as_ptr() as *mut f32, bias.as_ptr() as *const f32, m, n);
                }
            },
            DataType::F16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f16>(
                    a_ptr.as_ptr() as *const f16,
                    b_ptr.as_ptr() as *const f16,
                    ab_scale,
                    d_ptr.as_ptr() as *mut f16,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                    b_transpose,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<f16>(d_ptr.as_ptr() as *mut f16, bias.as_ptr() as *const f16, m, n);
                }
            },
            DataType::BF16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<bf16>(
                    a_ptr.as_ptr() as *const bf16,
                    b_ptr.as_ptr() as *const bf16,
                    ab_scale,
                    d_ptr.as_ptr() as *mut bf16,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                    b_transpose,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<bf16>(d_ptr.as_ptr() as *mut bf16, bias.as_ptr() as *const bf16, m, n);
                }
            },
            _ => unreachable!(),
        });
    }

    fn encode_quantized(
        &mut self,
        arguments: QuantizedMatmulArguments<Cpu>,
        configuration: &QuantizedMatmulConfiguration,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), QuantizedMatmulError<Cpu>> {
        if !matches!(
            configuration.data_type,
            DataType::F16 | DataType::BF16 | DataType::F32
        ) {
            return Err(QuantizedMatmulError::UnsupportedDataType(
                configuration.data_type,
            ));
        }
        if !matches!(configuration.group_size, 32 | 64 | 128) {
            return Err(QuantizedMatmulError::UnsupportedGroupSize(
                configuration.group_size,
            ));
        }

        if arguments.batch_dim >= 5 && configuration.output_dim > 1 {
            encode_quantized_gemm(encoder, arguments, configuration);
            return Ok(());
        }

        let bits = match configuration.mode {
            QuantizationMode::U4 => 4u32,
            QuantizationMode::I8 | QuantizationMode::U8 => 8u32,
        };
        let group_size = configuration.group_size as u32;
        let use_fast = configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0;

        let QuantizedMatmulArguments {
            a,
            a_offset,
            b,
            scales,
            zero_points_or_biases,
            output,
            hadamard_factors,
            batch_dim,
        } = arguments;

        let (zero_points, biases) = match configuration.quantization_method {
            QuantizationMethod::ScaleZeroPoint => (Some(zero_points_or_biases), None),
            QuantizationMethod::ScaleBias => (None, Some(zero_points_or_biases)),
        };

        let context = encoder.context();
        if use_fast {
            let kernel = <<Cpu as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                context,
                configuration.data_type,
                group_size,
                bits,
                configuration.quantization_method,
                configuration.use_hadamard,
            )
            .map_err(QuantizedMatmulError::BackendError)?;
            kernel.encode(
                b,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                output,
                hadamard_factors,
                configuration.input_dim as u32,
                configuration.output_dim as u32,
                batch_dim as u32,
                encoder,
            );
        } else {
            if configuration.use_hadamard {
                return Err(QuantizedMatmulError::UnsupportedHadamard);
            }
            let kernel = <<Cpu as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                context,
                configuration.data_type,
                group_size,
                bits,
                configuration.quantization_method,
            )
            .map_err(QuantizedMatmulError::BackendError)?;
            kernel.encode(
                b,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                output,
                configuration.input_dim as u32,
                configuration.output_dim as u32,
                batch_dim as u32,
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
