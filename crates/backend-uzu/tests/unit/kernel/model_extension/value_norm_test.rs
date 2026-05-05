use std::{
    fmt::{Debug, Display},
    ops::DerefMut,
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::ValueNormKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    qkv: Box<[T]>,
    batch_size: u32,
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
}

fn get_input<T: ArrayElement + Float>(
    batch_size: u32,
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
) -> Input<T> {
    let row_stride = ((num_heads + 2 * num_groups) * head_dim) as usize;
    let mut qkv = vec![T::zero(); batch_size as usize * row_stride];
    for (index, value) in qkv.iter_mut().enumerate() {
        *value = T::from(((index as f32) * 0.037 + 0.25).sin() * 3.0).unwrap();
    }

    Input {
        qkv: qkv.into_boxed_slice(),
        batch_size,
        num_heads,
        num_groups,
        head_dim,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::ValueNormKernel::new(&context, T::data_type())
        .expect("Failed to create ValueNormKernel");
    let qkv_array = context.create_array_from(&[input.qkv.len()], &input.qkv, "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        qkv_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.num_heads,
        input.num_groups,
        input.head_dim,
        1e-6,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    qkv_array.as_slice().to_vec()
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(input: &Input<T>) {
    let expected = get_output::<T, Cpu>(input);
    let epsilon = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(input);
        let message = format!(
            "ValueNorm failed for backend={}, batch_size={}, num_heads={}, num_groups={}, head_dim={}",
            std::any::type_name::<B>(),
            input.batch_size,
            input.num_heads,
            input.num_groups,
            input.head_dim,
        );
        assert_eq_float::<T>(&expected, &output, epsilon, &message);
    });
}

fn test_value_norm_model_extension_shape<T: ArrayElement + Float + Debug + Display>() {
    let input = get_input::<T>(20, 8, 1, 512);
    test_internal(&input);
}

#[uzu_test]
fn test_value_norm_model_extension_shape_f32() {
    test_value_norm_model_extension_shape::<f32>();
}

#[uzu_test]
fn test_value_norm_model_extension_shape_f16() {
    test_value_norm_model_extension_shape::<f16>();
}

#[uzu_test]
fn test_value_norm_model_extension_shape_bf16() {
    test_value_norm_model_extension_shape::<bf16>();
}
