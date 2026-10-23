use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, Kernels, kernel::SoftmaxKernel},
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_parity,
        helpers::{
            buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend, submit_command_buffer,
        },
    },
};

fn get_output<T: DataTypeRepr + Float, B: Backend>(
    values: &[T],
    sinks: Option<&[T]>,
    row_length: u32,
    outer_dim: u32,
    batch_dim: u32,
) -> Vec<T> {
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::SoftmaxKernel::new(&context, T::data_type(), sinks.is_some())
        .expect("Failed to create SoftmaxKernel");
    let mut values = create_buffer_with_data::<B, T>(&context, values);
    let sinks = sinks.map(|sinks| create_buffer_with_data::<B, T>(&context, sinks));
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(&mut values, sinks.as_ref(), row_length, outer_dim, batch_dim, &mut command_buffer);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, T>(&context, &values)
}

fn test_softmax<T: DataTypeRepr + Float>(has_sinks: bool) {
    let (rel_tol, abs_tol) = if matches!(T::data_type(), DataType::BF16) {
        (1e-2, 1e-6)
    } else {
        (1e-4, 1e-8)
    };
    // Rows within one workgroup's stride of 1024 elements and spanning several, with a partial tail.
    for (row_length, outer_dim, batch_dim) in [(7, 3, 2), (1024, 2, 3), (4099, 3, 2)] {
        let values = (0..row_length * outer_dim * batch_dim)
            .map(|i| T::from((i as f32 * 0.37).sin() * 8.0).unwrap())
            .collect::<Box<[_]>>();
        // Sinks below, within and above the row maximum.
        let sinks =
            has_sinks.then(|| (0..outer_dim).map(|i| T::from(i as f32 * 6.0 - 2.0).unwrap()).collect::<Box<[_]>>());
        let expected = get_output::<T, Cpu>(&values, sinks.as_deref(), row_length, outer_dim, batch_dim);
        for_each_non_cpu_backend!(|B| {
            let actual = get_output::<T, B>(&values, sinks.as_deref(), row_length, outer_dim, batch_dim);
            let label =
                format!("Softmax backend={} row_length={row_length} has_sinks={has_sinks}", std::any::type_name::<B>());
            assert_parity::<T>(&label, &expected, &actual, rel_tol, abs_tol);
        });
    }
}

#[uzu_test]
fn test_softmax_f32() {
    test_softmax::<f32>(false);
}

#[uzu_test]
fn test_softmax_bf16() {
    test_softmax::<bf16>(false);
}

#[uzu_test]
fn test_softmax_sinks_f32() {
    test_softmax::<f32>(true);
}

#[uzu_test]
fn test_softmax_sinks_bf16() {
    test_softmax::<bf16>(true);
}
