use half::{bf16, f16};

use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            kernel::matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
        },
        cpu::{Cpu, context::CpuContext},
    },
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
        _context: &CpuContext,
        arguments: MatmulArguments<Cpu>,
        encoder: &mut Encoder<Cpu>,
    ) {
        let command_buffer = encoder.as_command_buffer_mut();
        let m = arguments.batch_dim as usize;
        let n = arguments.output_dim as usize;
        let k = arguments.input_dim as usize;
        let lda = k;
        let ldb = k;
        let ldd = n;
        let a_offset = arguments.a_offset as usize;
        let data_type = self.data_type;

        let a_ptr = unsafe { &*arguments.a.get() }.as_ptr().wrapping_byte_add(a_offset);
        let b_ptr = unsafe { &*arguments.b.get() }.as_ptr();
        let ab_scale = arguments.ab_scale;
        let d_ptr = arguments.d.get_mut().as_mut_ptr();

        let (is_accumulate, bias_ptr) = match arguments.c {
            MatmulArgumentC::Accumulate => (true, None),
            MatmulArgumentC::Bias(bias) => (false, Some(unsafe { &*bias.get() }.as_ptr())),
            MatmulArgumentC::None => (false, None),
        };

        command_buffer.push_command(move || match data_type {
            DataType::F32 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f32>(
                    a_ptr as *const f32,
                    b_ptr as *const f32,
                    ab_scale,
                    d_ptr as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<f32>(d_ptr as *mut f32, bias as *const f32, m, n);
                }
            },
            DataType::F16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f16>(
                    a_ptr as *const f16,
                    b_ptr as *const f16,
                    ab_scale,
                    d_ptr as *mut f16,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<f16>(d_ptr as *mut f16, bias as *const f16, m, n);
                }
            },
            DataType::BF16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<bf16>(
                    a_ptr as *const bf16,
                    b_ptr as *const bf16,
                    ab_scale,
                    d_ptr as *mut bf16,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<bf16>(d_ptr as *mut bf16, bias as *const bf16, m, n);
                }
            },
            _ => unreachable!(),
        });
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
