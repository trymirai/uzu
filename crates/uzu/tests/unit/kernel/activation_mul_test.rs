use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::MlpGateActMulKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    fused_up: Box<[T]>,
    h: i32,
    m: i32,
    act_type: ActivationType,
}

fn get_test_data<T: ArrayElement + Float>(act_type: ActivationType) -> (Input<T>, Vec<T>) {
    let h = 64i32;
    let m = 4i32;
    let fused_len = (m * 2 * h) as usize;

    let mut fused_up: Vec<T> = vec![T::zero(); fused_len];
    for i in 0..fused_len {
        fused_up[i] = T::from((i as f32 * 0.1).sin() * 2.0f32).unwrap();
    }

    let input = Input {
        fused_up: fused_up.into_boxed_slice(),
        h,
        m,
        act_type,
    };

    let expected = get_output::<T, Cpu>(&input);

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::MlpGateActMulKernel::new(&context, T::data_type())
        .expect("Failed to create MlpGateActMulKernel");

    let fused_len = (input.m * 2 * input.h) as usize;
    let out_len = (input.m * input.h) as usize;
    let fused_up_array = context.create_array_from(&[fused_len], &input.fused_up, "");
    let hidden_array = context.create_array_uninitialized(&[out_len], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        fused_up_array.buffer().borrow().deref(),
        hidden_array.buffer().borrow_mut().deref_mut(),
        input.h,
        input.m,
        input.act_type,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    hidden_array.as_slice().to_vec()
}

fn test<T: ArrayElement + Float + Debug + Display>(act_type: ActivationType) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(act_type);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Results are not equal for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

#[test]
fn test_silu_f32() {
    test::<f32>(ActivationType::silu_default());
}

#[test]
fn test_silu_f16() {
    test::<f16>(ActivationType::silu_default());
}

#[test]
fn test_silu_bf16() {
    test::<bf16>(ActivationType::silu_default());
}

#[test]
fn test_gelu_f32() {
    test::<f32>(ActivationType::GELU);
}

#[test]
fn test_gelu_f16() {
    test::<f16>(ActivationType::GELU);
}

#[test]
fn test_gelu_bf16() {
    test::<bf16>(ActivationType::GELU);
}
