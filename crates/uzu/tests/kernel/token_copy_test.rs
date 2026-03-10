use std::ops::{Deref, DerefMut};

use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending, Context,
        Kernels,
        kernel::{TokenCopySampledKernel, TokenCopyToResultsKernel},
    },
};

fn test_token_copy_sampled_impl<B: Backend>(src_value: u32) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TokenCopySampledKernel::new(&context)
        .expect("Failed to create TokenCopySampledKernel");

    let src_array = context.create_array_from(&[1], &[src_value], "");
    let dst_array = context.create_array_uninitialized(&[1], u64::data_type(), "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        src_array.buffer().borrow().deref(),
        dst_array.buffer().borrow_mut().deref_mut(),
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let output = dst_array.as_slice::<u64>().to_vec();
    assert_eq!(output[0], src_value as u64, "TokenCopySampled failed for backend {}", std::any::type_name::<B>());
}

fn test_token_copy_to_results_impl<B: Backend>(src_value: u32) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel::new(&context)
        .expect("Failed to create TokenCopyToResultsKernel");

    let src_array = context.create_array_from(&[1], &[src_value], "");
    let dst_array = context.create_array_uninitialized(&[1], u32::data_type(), "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        src_array.buffer().borrow().deref(),
        dst_array.buffer().borrow_mut().deref_mut(),
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let output = dst_array.as_slice::<u32>().to_vec();
    assert_eq!(output[0], src_value, "TokenCopyToResults failed for backend {}", std::any::type_name::<B>());
}

#[test]
fn test_token_copy_sampled() {
    for_each_backend!(|B| {
        test_token_copy_sampled_impl::<B>(0);
        test_token_copy_sampled_impl::<B>(42);
        test_token_copy_sampled_impl::<B>(u32::MAX);
    });
}

#[test]
fn test_token_copy_to_results() {
    for_each_backend!(|B| {
        test_token_copy_to_results_impl::<B>(0);
        test_token_copy_to_results_impl::<B>(42);
        test_token_copy_to_results_impl::<B>(u32::MAX);
    });
}
