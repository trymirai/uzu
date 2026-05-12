use backend_uzu::{
    backends::common::{
        Backend, Context, Encoder, Kernels,
        kernel::{TokenCopySampledKernel, TokenCopyToResultsKernel},
    },
};

use crate::{
    common::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    uzu_test,
};

fn test_token_copy_sampled_impl<B: Backend>(src_value: u32) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TokenCopySampledKernel::new(&context)
        .expect("Failed to create TokenCopySampledKernel");

    let src_allocation = alloc_allocation_with_data::<B, u32>(&context, &[src_value]);
    let mut dst_allocation = alloc_allocation::<B, u64>(&context, 1);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&src_allocation, &mut dst_allocation, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output = allocation_to_vec::<B, u64>(&dst_allocation);
    assert_eq!(output[0], src_value as u64, "TokenCopySampled failed for backend {}", std::any::type_name::<B>());
}

fn test_token_copy_to_results_impl<B: Backend>(src_value: u32) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel::new(&context)
        .expect("Failed to create TokenCopyToResultsKernel");

    let src_allocation = alloc_allocation_with_data::<B, u32>(&context, &[src_value]);
    let mut dst_allocation = alloc_allocation::<B, u32>(&context, 1);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&src_allocation, &mut dst_allocation, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output = allocation_to_vec::<B, u32>(&dst_allocation);
    assert_eq!(output[0], src_value, "TokenCopyToResults failed for backend {}", std::any::type_name::<B>());
}

#[uzu_test]
fn test_token_copy_sampled() {
    for_each_backend!(|B| {
        test_token_copy_sampled_impl::<B>(0);
        test_token_copy_sampled_impl::<B>(42);
        test_token_copy_sampled_impl::<B>(u32::MAX);
    });
}

#[uzu_test]
fn test_token_copy_to_results() {
    for_each_backend!(|B| {
        test_token_copy_to_results_impl::<B>(0);
        test_token_copy_to_results_impl::<B>(42);
        test_token_copy_to_results_impl::<B>(u32::MAX);
    });
}
