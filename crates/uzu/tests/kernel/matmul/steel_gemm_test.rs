use std::fmt::{Debug, Display};

use half::{bf16, f16};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use num_traits::Float;
use uzu::{
    ArrayElement,
    backends::{
        common::{
            CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending, Context,
            kernel::matmul::MatmulArguments,
        },
        metal::{MetalContext, kernel::matmul::MatmulMetalKernel},
    },
};

use crate::common::assert::assert_eq_float;

fn reference_gemm<T: ArrayElement + Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut output = vec![T::zero(); m * n];
    for row in 0..m {
        for col in 0..n {
            let mut accumulator = 0.0f64;
            for i in 0..k {
                accumulator += a[row * k + i].to_f64().unwrap() * b[col * k + i].to_f64().unwrap();
            }
            output[row * n + col] = T::from(accumulator).unwrap();
        }
    }
    output
}

fn to_bytes<T: ArrayElement>(data: &[T]) -> Vec<u8> {
    let byte_length = data.len() * std::mem::size_of::<T>();
    let mut bytes = vec![0u8; byte_length];
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, bytes.as_mut_ptr(), byte_length);
    }
    bytes
}

fn read_output<T: ArrayElement + Float>(buffer: &objc2::runtime::ProtocolObject<dyn metal::MTLBuffer>, count: usize) -> Vec<T> {
    let pointer = buffer.contents().as_ptr() as *const T;
    unsafe { std::slice::from_raw_parts(pointer, count).to_vec() }
}

fn test_gemm<T: ArrayElement + Float + Debug + Display>(m: usize, k: usize, n: usize, tolerance: f32) {
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let expected = reference_gemm(&a, &b, m, k, n);

    let context = MetalContext::new().expect("Metal context required");

    let a_buffer = context.device.new_buffer_with_data(&to_bytes(&a), MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
    let b_buffer = context.device.new_buffer_with_data(&to_bytes(&b), MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
    let mut d_buffer = context.device.new_buffer(m * n * T::data_type().size_in_bytes(), MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();

    let mut kernel = <MatmulMetalKernel as uzu::backends::common::kernel::matmul::MatmulKernel>::new(&context, T::data_type()).expect("kernel creation");
    let mut command_buffer = context.create_command_buffer().unwrap().start_encoding();

    kernel.encode_gemm(
        &context,
        &mut command_buffer,
        MatmulArguments {
            a: &a_buffer,
            a_offset: 0,
            b: &b_buffer,
            output: &mut d_buffer,
            bias: None,
            batch: m,
            input_dim: k,
            output_dim: n,
        },
    );

    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let actual: Vec<T> = read_output(&d_buffer, m * n);
    assert_eq_float(&expected, &actual, tolerance, "Metal GEMM");
}

#[test]
fn test_f32_aligned() { test_gemm::<f32>(64, 64, 64, 0.01); }
#[test]
fn test_f16_aligned() { test_gemm::<f16>(64, 64, 64, 0.01); }
#[test]
fn test_bf16_aligned() { test_gemm::<bf16>(64, 64, 64, 0.1); }
#[test]
fn test_f32_unaligned() { test_gemm::<f32>(7, 33, 11, 0.01); }
#[test]
fn test_f16_unaligned() { test_gemm::<f16>(7, 33, 11, 0.01); }
#[test]
fn test_bf16_unaligned() { test_gemm::<bf16>(7, 33, 11, 0.1); }
#[test]
fn test_f32_large() { test_gemm::<f32>(16, 128, 256, 0.01); }
#[test]
fn test_f16_large() { test_gemm::<f16>(16, 128, 256, 0.01); }
#[test]
fn test_bf16_large() { test_gemm::<bf16>(16, 128, 256, 0.1); }
