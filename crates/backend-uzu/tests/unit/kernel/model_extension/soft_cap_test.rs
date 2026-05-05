use std::ops::{Deref, DerefMut};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::SoftCapKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

fn get_output<B: Backend>(
    input: &[f32],
    cap: f32,
    in_place: bool,
) -> Vec<f32> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::SoftCapKernel::new(&context, f32::data_type(), in_place)
        .expect("Failed to create SoftCapKernel");

    let input_array = context.create_array_from(&[input.len()], input, "soft_cap_input");
    let output_array = if in_place {
        context.create_array_from(&[input.len()], input, "soft_cap_output")
    } else {
        context.create_array_uninitialized(&[input.len()], f32::data_type(), "soft_cap_output")
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let input_buffer = input_array.buffer();
    let output_buffer = output_array.buffer();
    let input_borrow = input_buffer.borrow();
    let mut output_borrow = output_buffer.borrow_mut();
    let input_arg = if in_place {
        None
    } else {
        Some(input_borrow.deref())
    };
    kernel.encode(input_arg, output_borrow.deref_mut(), input.len() as u32, cap, &mut encoder);
    drop(output_borrow);
    drop(input_borrow);
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    output_array.as_slice().to_vec()
}

fn test_internal(in_place: bool) {
    let input = [-90.0, -30.0, -3.0, 0.0, 2.0, 30.0, 90.0];
    let expected = get_output::<Cpu>(&input, 30.0, in_place);

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B>(&input, 30.0, in_place);
        assert_eq_float(&expected, &output, 1e-5, "SoftCap output mismatch");
    });
}

#[uzu_test]
fn test_soft_cap_out_of_place() {
    test_internal(false);
}

#[uzu_test]
fn test_soft_cap_in_place() {
    test_internal(true);
}
