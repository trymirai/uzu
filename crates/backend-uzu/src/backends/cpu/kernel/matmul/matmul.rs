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
        let ldb = k;
        let ldd = n;
        let data_type = self.data_type;
        let (a, a_range) = a.as_buffer_range();
        let (b, b_range) = b.as_buffer_range();
        let (d, d_range) = d.as_buffer_range();
        let a_offset = a_range.start + a_offset;
        let b_offset = b_range.start;
        let d_offset = d_range.start;

        let a_ptr = SendPtr(unsafe { &*a.get() }.as_ptr().wrapping_byte_add(a_offset));
        let b_ptr = SendPtr(unsafe { &*b.get() }.as_ptr().wrapping_byte_add(b_offset));
        let d_ptr = SendPtrMut(unsafe { (&*d.get()).as_ptr().wrapping_byte_add(d_offset) as *mut u8 });

        let (is_accumulate, bias_ptr) = match c {
            MatmulArgumentC::Accumulate => (true, None),
            MatmulArgumentC::Bias(bias) => {
                let (bias, bias_range) = bias.as_buffer_range();
                (false, Some(SendPtr(unsafe { &*bias.get() }.as_ptr().wrapping_byte_add(bias_range.start))))
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
                    is_accumulate,
                );
                if let Some(bias) = bias_ptr {
                    apply_bias::<bf16>(d_ptr.as_ptr() as *mut bf16, bias.as_ptr() as *const bf16, m, n);
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
