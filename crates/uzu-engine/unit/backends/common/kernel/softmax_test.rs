use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::SoftmaxKernel,
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_non_cpu_backend},
};

/// One in-place Softmax dispatch on backend `B` over `outer_dim * batch_dim` rows.
fn get_output<T: ArrayElement, B: Backend>(
    values: &[T],
    sinks: Option<&[T]>,
    row_length: u32,
    outer_dim: u32,
    batch_dim: u32,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::SoftmaxKernel::new(&context, T::data_type(), sinks.is_some())
        .expect("Failed to create SoftmaxKernel");
    let mut values_buffer = create_buffer_with_data::<B, T>(&context, values);
    let sinks_buffer = sinks.map(|sinks| create_buffer_with_data::<B, T>(&context, sinks));
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(&mut values_buffer, sinks_buffer.as_ref(), row_length, outer_dim, batch_dim, &mut command_buffer);
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<B, T>(&values_buffer)
}

/// Rows of -FLT_MAX are finite and must normalize to `1 / n`, or `1 / (n + 1)` with an equal finite sink (an
/// omitted virtual output). Padded tail lanes of the last chunk used to add to the normalizer on Metal.
#[uzu_test]
fn lowest_finite_rows_ignore_padded_tails() {
    for row_length in [1usize, 3, 31, 255, 256, 257] {
        for sinks in [None, Some([-f32::MAX; 2])] {
            let values = vec![-f32::MAX; row_length * 4];
            let sinks = sinks.as_ref().map(|sinks| &sinks[..]);
            let expected = 1.0 / (row_length + usize::from(sinks.is_some())) as f32;
            let case = format!("row {row_length} sinks {}", sinks.is_some());
            let check = |backend: &str, output: &[f32]| {
                for (index, &value) in output.iter().enumerate() {
                    assert!((value - expected).abs() <= 1e-6 * expected, "{backend} {case} element {index}: {value}");
                }
                for row in output.chunks(row_length) {
                    let mass = row.iter().sum::<f32>();
                    assert!((mass - row_length as f32 * expected).abs() <= 1e-5, "{backend} {case} row mass {mass}");
                }
            };
            check("CPU", &get_output::<f32, Cpu>(&values, sinks, row_length as u32, 2, 2));
            for_each_non_cpu_backend!(|B| {
                check(std::any::type_name::<B>(), &get_output::<f32, B>(&values, sinks, row_length as u32, 2, 2));
            });
        }
    }
}
