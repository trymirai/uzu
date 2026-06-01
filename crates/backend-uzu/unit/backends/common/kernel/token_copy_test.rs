use proc_macros::uzu_test;
use test_runner::for_each_backend;

use crate::{
    backends::common::{Allocation, Backend, Context, Encoder, Kernels, kernel::TokenCopySampledKernel},
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

fn test_token_copy_sampled_impl<B: Backend>(src_value: u32) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TokenCopySampledKernel::new(&context, false)
        .expect("Failed to create TokenCopySampledKernel");

    let src_allocation = alloc_allocation_with_data::<B, u32>(&context, &[src_value]);
    let mut dst_allocation = alloc_allocation::<B, u64>(&context, 1);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&src_allocation, &mut dst_allocation, None::<&mut Allocation<B>>, None, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output = allocation_to_vec::<B, u64>(&dst_allocation);
    assert_eq!(output[0], src_value as u64, "TokenCopySampled failed for backend {}", std::any::type_name::<B>());
}

#[uzu_test]
fn test_token_copy_sampled() {
    for_each_backend!(|B| {
        test_token_copy_sampled_impl::<B>(0);
        test_token_copy_sampled_impl::<B>(42);
        test_token_copy_sampled_impl::<B>(u32::MAX);
    });
}
