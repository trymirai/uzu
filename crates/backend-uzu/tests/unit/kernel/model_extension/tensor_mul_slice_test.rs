use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::TensorMulSliceKernel},
        cpu::Cpu,
    },
};
use std::ops::{Deref, DerefMut};

use crate::{common::assert::assert_eq_float, uzu_test};

fn get_output<B: Backend>() -> Vec<f32> {
    let suffix_length = 3u32;
    let total_slice_dim = 8u32;
    let slice_dim = 2u32;
    let slice_index = 2u32;
    let values: [f32; 6] = [1.0, 2.0, -3.0, 4.0, 0.5, -0.25];
    let slice_source: [f32; 24] = [
        1.0, 1.0, 2.0, 2.0, 10.0, 20.0, 3.0, 3.0, -1.0, -1.0, 4.0, 4.0, 30.0, 40.0, 5.0, 5.0,
        6.0, 6.0, 7.0, 7.0, 50.0, 60.0, 8.0, 8.0,
    ];

    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::TensorMulSliceKernel::new(&context, f32::data_type())
        .expect("Failed to create TensorMulSliceKernel");
    let values_array = context.create_array_from(&[values.len()], &values, "values");
    let slice_source_array = context.create_array_from(&[slice_source.len()], &slice_source, "slice_source");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        values_array.buffer().borrow_mut().deref_mut(),
        slice_source_array.buffer().borrow().deref(),
        suffix_length,
        total_slice_dim,
        slice_dim,
        slice_index,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    values_array.as_slice().to_vec()
}

#[uzu_test]
fn test_tensor_mul_slice() {
    let expected = get_output::<Cpu>();

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B>();
        assert_eq_float(&expected, &output, 1e-5, "TensorMulSlice output mismatch");
    });
}
