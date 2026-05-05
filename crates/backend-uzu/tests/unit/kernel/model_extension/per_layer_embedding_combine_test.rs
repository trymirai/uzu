use std::ops::{Deref, DerefMut};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::PerLayerEmbeddingCombineKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

fn get_output<B: Backend>() -> Vec<f32> {
    let suffix_length = 2u32;
    let num_layers = 3u32;
    let ple_dim = 4u32;
    let total_ple_dim = (num_layers * ple_dim) as usize;
    let token_ple = (0..suffix_length as usize * total_ple_dim)
        .map(|index| index as f32 * 0.03125 - 0.5)
        .collect::<Vec<_>>();
    let model_ple = (0..suffix_length as usize * total_ple_dim)
        .map(|index| (index as f32 * 0.17).sin())
        .collect::<Vec<_>>();
    let scales = [0.25, 0.5, 0.75, 1.0];

    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::PerLayerEmbeddingCombineKernel::new(
        &context,
        f32::data_type(),
        f32::data_type(),
    )
    .expect("Failed to create PerLayerEmbeddingCombineKernel");
    let token_ple_array = context.create_array_from(&[token_ple.len()], &token_ple, "token_ple");
    let model_ple_array = context.create_array_from(&[model_ple.len()], &model_ple, "model_ple");
    let scales_array = context.create_array_from(&[scales.len()], &scales, "scales");
    let combined_array = context.create_array_uninitialized(&[token_ple.len()], f32::data_type(), "combined");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        token_ple_array.buffer().borrow().deref(),
        model_ple_array.buffer().borrow().deref(),
        scales_array.buffer().borrow().deref(),
        combined_array.buffer().borrow_mut().deref_mut(),
        suffix_length,
        num_layers,
        ple_dim,
        0.25,
        0.70710677,
        1e-6,
        0.0,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    combined_array.as_slice().to_vec()
}

#[uzu_test]
fn test_per_layer_embedding_combine() {
    let expected = get_output::<Cpu>();

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B>();
        assert_eq_float(&expected, &output, 1e-5, "PerLayerEmbeddingCombine output mismatch");
    });
}
