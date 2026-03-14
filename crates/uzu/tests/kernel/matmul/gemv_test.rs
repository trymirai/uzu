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

fn reference_gemv<T: ArrayElement + Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
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

fn test_gemv<T: ArrayElement + Float + Debug + Display>(m: usize, k: usize, n: usize, tolerance: f32) {
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let expected = reference_gemv(&a, &b, m, k, n);

    let context = MetalContext::new().expect("Metal context required");

    let a_buffer = context.device.new_buffer_with_data(&to_bytes(&a), MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
    let b_buffer = context.device.new_buffer_with_data(&to_bytes(&b), MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
    let mut d_buffer = context.device.new_buffer(m * n * T::data_type().size_in_bytes(), MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();

    let mut kernel = <MatmulMetalKernel as uzu::backends::common::kernel::matmul::MatmulKernel>::new(&context, T::data_type()).expect("kernel creation");
    let mut command_buffer = context.create_command_buffer().unwrap().start_encoding();

    kernel.encode_gemv(
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
    assert_eq_float(&expected, &actual, tolerance, "Metal GEMV");
}

#[test]
fn test_f32_m1() { test_gemv::<f32>(1, 128, 64, 0.01); }
#[test]
fn test_f16_m1() { test_gemv::<f16>(1, 128, 64, 0.01); }
#[test]
fn test_bf16_m1() { test_gemv::<bf16>(1, 128, 64, 0.1); }
#[test]
fn test_f32_batched() { test_gemv::<f32>(4, 128, 64, 0.01); }
#[test]
fn test_f16_batched() { test_gemv::<f16>(4, 128, 64, 0.01); }
#[test]
fn test_bf16_batched() { test_gemv::<bf16>(4, 128, 64, 0.1); }
#[test]
fn test_f32_max_batch() { test_gemv::<f32>(8, 128, 64, 0.01); }
#[test]
fn test_f16_max_batch() { test_gemv::<f16>(8, 128, 64, 0.01); }
#[test]
fn test_bf16_max_batch() { test_gemv::<bf16>(8, 128, 64, 0.1); }
#[test]
fn test_f32_unaligned_k() { test_gemv::<f32>(1, 33, 64, 0.01); }
#[test]
fn test_f16_unaligned_k() { test_gemv::<f16>(1, 33, 64, 0.01); }
#[test]
fn test_bf16_unaligned_k() { test_gemv::<bf16>(1, 33, 64, 0.1); }
#[test]
fn test_f32_unaligned_n() { test_gemv::<f32>(1, 128, 11, 0.01); }
#[test]
fn test_f16_unaligned_n() { test_gemv::<f16>(1, 128, 11, 0.01); }
#[test]
fn test_bf16_unaligned_n() { test_gemv::<bf16>(1, 128, 11, 0.1); }
#[test]
fn test_f32_large() { test_gemv::<f32>(1, 4096, 2048, 0.05); }
#[test]
fn test_f16_large() { test_gemv::<f16>(1, 4096, 2048, 0.5); }
#[test]
fn test_bf16_large() { test_gemv::<bf16>(1, 4096, 2048, 1.0); }
#[test]
fn test_f32_small_n() { test_gemv::<f32>(1, 128, 3, 0.01); }
#[test]
fn test_f16_small_n() { test_gemv::<f16>(1, 128, 3, 0.01); }
#[test]
fn test_bf16_small_n() { test_gemv::<bf16>(1, 128, 3, 0.1); }
