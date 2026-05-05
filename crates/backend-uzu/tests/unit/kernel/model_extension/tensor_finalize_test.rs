use std::ops::DerefMut;

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::TensorFinalizeKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

fn get_output<B: Backend>(has_scalar: bool) -> (Vec<f32>, Vec<f32>) {
    let shortcut: [f32; 4] = [1.0, -2.0, 3.0, -4.0];
    let main: [f32; 4] = [0.5, 1.5, -2.0, 4.0];
    let scalar: [f32; 1] = [0.25];

    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::TensorFinalizeKernel::new(
        &context,
        f32::data_type(),
        has_scalar,
    )
    .expect("Failed to create TensorFinalizeKernel");
    let shortcut_array = context.create_array_from(&[shortcut.len()], &shortcut, "shortcut");
    let main_array = context.create_array_from(&[main.len()], &main, "main");
    let scalar_array = has_scalar.then(|| context.create_array_from(&[1], &scalar, "scalar"));

    let scalar_buffer = scalar_array.as_ref().map(|array| array.buffer());
    let scalar_borrow = scalar_buffer.as_ref().map(|buffer| buffer.borrow());
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        shortcut_array.buffer().borrow_mut().deref_mut(),
        main_array.buffer().borrow_mut().deref_mut(),
        scalar_borrow.as_deref(),
        shortcut.len() as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    (shortcut_array.as_slice().to_vec(), main_array.as_slice().to_vec())
}

fn test_internal(has_scalar: bool) {
    let (expected_shortcut, expected_main) = get_output::<Cpu>(has_scalar);

    for_each_non_cpu_backend!(|B| {
        let (shortcut, main) = get_output::<B>(has_scalar);
        assert_eq_float(&expected_shortcut, &shortcut, 1e-5, "TensorFinalize shortcut mismatch");
        assert_eq_float(&expected_main, &main, 1e-5, "TensorFinalize main mismatch");
    });
}

#[uzu_test]
fn test_tensor_finalize_without_scalar() {
    test_internal(false);
}

#[uzu_test]
fn test_tensor_finalize_with_scalar() {
    test_internal(true);
}
