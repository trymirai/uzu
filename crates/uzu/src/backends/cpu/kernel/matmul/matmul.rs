use half::{bf16, f16};

use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel},
        cpu::{Cpu, command_buffer::CpuCommandBuffer, context::CpuContext},
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
        command_buffer: &mut CpuCommandBuffer,
    ) {
        let m = arguments.batch as usize;
        let n = arguments.output_dim as usize;
        let k = arguments.input_dim as usize;
        let lda = arguments.leading_dimension_a as usize;
        let ldb = arguments.leading_dimension_b as usize;
        let ldd = arguments.leading_dimension_d as usize;
        let a_offset = arguments.a_offset as usize;
        let data_type = self.data_type;

        let a_ptr = arguments.a.as_ptr().wrapping_byte_add(a_offset);
        let b_ptr = arguments.b.as_ptr();
        let d_ptr = arguments.d.as_mut_ptr();
        let bias_ptr = arguments.bias.map(|b| b.as_ptr());

        command_buffer.push_command(move || match data_type {
            DataType::F32 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f32>(
                    a_ptr as *const f32,
                    b_ptr as *const f32,
                    d_ptr as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<f32>(d_ptr as *mut f32, bias as *const f32, m, n);
                }
            },
            DataType::F16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<f16>(
                    a_ptr as *const f16,
                    b_ptr as *const f16,
                    d_ptr as *mut f16,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<f16>(d_ptr as *mut f16, bias as *const f16, m, n);
                }
            },
            DataType::BF16 => {
                super::gemm::shaders::steel_gemm::matmul_gemm_impl::<bf16>(
                    a_ptr as *const bf16,
                    b_ptr as *const bf16,
                    d_ptr as *mut bf16,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldd,
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
