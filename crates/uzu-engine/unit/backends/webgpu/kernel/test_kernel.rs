use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            CommandBufferEncoding, Context,
            gpu_types::{
                activation_type::ActivationType, kv_cache_update::Copy, quantization::QuantizationMode,
                ring::RingParams,
            },
        },
        webgpu::{
            WebGPU,
            buffer::{binding_parts, dense::WebGPUDenseBufferGpu},
            command_buffer::WebGPUCommandBufferEncoding,
            context::WebGPUContext,
            error::WebGPUError,
        },
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer_with_data, submit_command_buffer},
};

include!(concat!(env!("OUT_DIR"), "/webgpu/test_kernel.rs"));

fn data<T: DataTypeRepr>(count: usize) -> Vec<T> {
    (0..count).map(|i| T::from((i % 31) as f32 / 4.0 - 3.75).unwrap()).collect()
}

fn filled<T: DataTypeRepr>(
    count: usize,
    value: f32,
) -> Vec<T> {
    vec![T::from(value).unwrap(); count]
}

fn check<T: DataTypeRepr>(
    context: &WebGPUContext,
    initial: &[T],
    expected: &[T],
    tolerance: f32,
    encode: impl FnOnce(&mut WebGPUDenseBufferGpu, &mut WebGPUCommandBufferEncoding),
) {
    let mut output = create_buffer_with_data::<WebGPU, T>(context, initial);
    let mut commands = context.create_command_buffer(None, None).unwrap();
    encode(&mut output, &mut commands);
    submit_command_buffer(commands);
    let actual = buffer_to_vec::<WebGPU, T>(context, &output);
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.to_f64().unwrap();
        let expected = expected.to_f64().unwrap();
        assert!((actual - expected).abs() <= tolerance as f64, "element {i}: expected {expected}, got {actual}");
    }
}

fn dispatch_1d<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let group_thread = TestKernelGroupThreadWebGPUKernel::new(&context, T::data_type()).unwrap();
    for size in [1usize, 255, 256, 257, 513] {
        let initial = filled::<T>((size + 3).next_multiple_of(2), -100.0);
        let a = data::<T>(initial.len());
        let b = filled::<T>(initial.len(), 2.0);
        let input_0 = create_buffer_with_data::<WebGPU, T>(&context, &a);
        let input_1 = create_buffer_with_data::<WebGPU, T>(&context, &b);
        let mut expected = initial.clone();
        for (dst, src) in expected[..size].iter_mut().zip(&a) {
            *dst = T::from(src.to_f32().unwrap() + 2.0).unwrap();
        }
        check(&context, &initial, &expected, 0.0, |output, commands| {
            group_thread.encode(&input_0, &input_1, output, size as u32, commands);
        });
        for bs in [2, 4, 8] {
            let kernel = TestKernelAxisWebGPUKernel::new(&context, T::data_type(), bs).unwrap();
            check(&context, &initial, &expected, 0.0, |output, commands| {
                kernel.encode(&input_0, &input_1, output, size as u32, commands);
            });
        }
        if T::data_type() == DataType::F32 {
            let kernel = TestKernelAxisFloatWebGPUKernel::new(&context).unwrap();
            check(&context, &initial, &expected, 0.0, |output, commands| {
                kernel.encode(&input_0, &input_1, output, size as u32, commands);
            });
        }
    }
}

fn dispatch_2d<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let rows = 35;
    let cols = 67;
    let count = rows * cols;
    let values = data::<T>(count + 3);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let initial = filled::<T>(values.len(), -100.0);
    let mut expected = initial.clone();
    expected[..count].copy_from_slice(&values[..count]);
    let axis = TestKernelAxis2dWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        axis.encode(&input, output, rows as u32, cols as u32, commands);
    });
    let groups = TestKernelGroups2dWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        groups.encode(&input, output, rows as u32, cols as u32, commands);
    });
    for bm in [16, 32, 64] {
        for bn in [16, 32, 64] {
            let kernel = TestKernelConstraintWebGPUKernel::new(&context, T::data_type(), bm, bn);
            if bm * bn > 1024 {
                assert!(matches!(kernel, Err(WebGPUError::UnsupportedKernelVariant(_))));
            } else {
                let kernel = kernel.unwrap();
                check(&context, &initial, &expected, 0.0, |output, commands| {
                    kernel.encode(&input, output, rows as u32, cols as u32, commands);
                });
            }
        }
    }
}

fn multispecialize<I: DataTypeRepr, O: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let values = data::<I>(260);
    let input = create_buffer_with_data::<WebGPU, I>(&context, &values);
    let initial = filled::<O>(values.len(), -100.0);
    let mut expected = initial.clone();
    for (dst, src) in expected[..257].iter_mut().zip(&values) {
        *dst = O::from(src.to_f32().unwrap()).unwrap();
    }
    let kernel = TestKernelMultispecializeWebGPUKernel::new(&context, I::data_type(), O::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        kernel.encode(&input, output, 257, commands);
    });
}

fn specialize_and_optional<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let values = data::<T>(260);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let bias = create_buffer_with_data::<WebGPU, T>(&context, &filled::<T>(260, 2.0));
    let initial = filled::<T>(260, -100.0);
    for in_place in [false, true] {
        let mut expected = initial.clone();
        for (dst, src) in expected[..257].iter_mut().zip(&values) {
            *dst = if in_place {
                T::from(-99.0).unwrap()
            } else {
                *src
            };
        }
        let kernel = TestKernelSpecializeWebGPUKernel::new(&context, T::data_type(), in_place).unwrap();
        check(&context, &initial, &expected, 0.0, |output, commands| {
            kernel.encode(&input, output, 257, commands);
        });
    }
    for use_bias in [false, true] {
        let size = if use_bias {
            129
        } else {
            257
        };
        let mut expected = initial.clone();
        for (dst, src) in expected[..size].iter_mut().zip(&values) {
            *dst = T::from(
                src.to_f32().unwrap()
                    + if use_bias {
                        2.0
                    } else {
                        0.0
                    },
            )
            .unwrap();
        }
        let kernel = TestKernelOptionalWebGPUKernel::new(&context, T::data_type(), use_bias).unwrap();
        check(&context, &initial, &expected, 0.0, |output, commands| {
            kernel.encode(&input, use_bias.then_some(&bias), output, use_bias.then_some(size as u32), commands);
        });
    }
}

fn indirect<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let values = data::<T>(4 * 64);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let tile_map = create_buffer_with_data::<WebGPU, u32>(&context, &[3, 1, 0]);
    let dispatch = create_buffer_with_data::<WebGPU, u32>(&context, &[2, 1, 1]);
    let initial = filled::<T>(values.len(), -100.0);
    let mut expected = initial.clone();
    for tile in [3, 1] {
        expected[tile * 64..(tile + 1) * 64].copy_from_slice(&values[tile * 64..(tile + 1) * 64]);
    }
    let kernel = TestKernelIndirectWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        kernel.encode(&input, &tile_map, output, &dispatch, commands);
    });
}

fn constant_array<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let values = data::<T>(260);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let factors = std::array::from_fn(|i| i as i32 - 8);
    let initial = filled::<T>(values.len(), -100.0);
    let mut expected = initial.clone();
    for i in 0..257 {
        expected[i] = T::from(values[i].to_f32().unwrap() * factors[i % 16] as f32).unwrap();
    }
    let kernel = TestKernelConstantArrayWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        kernel.encode(&input, &factors, output, 257, commands);
    });
}

fn groupshared<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let batch_size = 4;
    let element_count = 259;
    let values = data::<T>(batch_size * element_count);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let initial = filled::<T>(batch_size, -100.0);
    let expected = values
        .chunks_exact(element_count)
        .map(|chunk| T::from(chunk.iter().map(|value| value.to_f32().unwrap()).sum::<f32>()).unwrap())
        .collect::<Vec<_>>();
    let kernel = TestKernelGroupsharedSizedWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        kernel.encode(&input, output, batch_size as u32, element_count as u32, commands);
    });

    let element_count = 67;
    let values = data::<T>(batch_size);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let initial = filled::<T>(batch_size * element_count, -100.0);
    let expected = values.iter().flat_map(|value| std::iter::repeat_n(*value, element_count)).collect::<Vec<_>>();
    let kernel = TestKernelGroupsharedRefWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        kernel.encode(&input, output, batch_size as u32, element_count as u32, commands);
    });
}

fn enums<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let values = data::<T>(260);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let initial = filled::<T>(values.len(), -100.0);
    for act in [
        ActivationType::SILU,
        ActivationType::GELUApprox,
        ActivationType::GELUExact,
        ActivationType::IDENTITY,
        ActivationType::SOFTPLUS,
    ] {
        let mut expected = initial.clone();
        for (dst, src) in expected[..257].iter_mut().zip(&values) {
            let x = src.to_f32().unwrap();
            *dst = T::from(if act == ActivationType::GELUExact {
                x
            } else {
                act.activate(x)
            })
            .unwrap();
        }
        let kernel = TestKernelEnumSpecializeWebGPUKernel::new(&context, T::data_type(), act).unwrap();
        let tolerance = if T::data_type() == DataType::F16 {
            0.004
        } else {
            1e-5
        };
        check(&context, &initial, &expected, tolerance, |output, commands| {
            kernel.encode(&input, output, 257, commands);
        });
    }
    let kernel = TestKernelEnumRuntimeWebGPUKernel::new(&context, T::data_type()).unwrap();
    for (mode, scale) in [(QuantizationMode::U4, 15.0), (QuantizationMode::I8, 127.0), (QuantizationMode::U8, 255.0)] {
        let mut expected = initial.clone();
        for (dst, src) in expected[..257].iter_mut().zip(&values) {
            *dst = T::from(src.to_f32().unwrap() * scale).unwrap();
        }
        check(&context, &initial, &expected, 0.0, |output, commands| {
            kernel.encode(&input, output, 257, mode, commands);
        });
    }
}

fn struct_constants<T: DataTypeRepr>() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let stride = 257;
    let values = data::<T>(4 * stride);
    let input = create_buffer_with_data::<WebGPU, T>(&context, &values);
    let initial = filled::<T>(values.len(), -100.0);
    let mut expected = initial.clone();
    expected[stride..2 * stride].copy_from_slice(&values[2 * stride..3 * stride]);
    let copy = Copy {
        source: 2,
        destination: 1,
    };
    let scalar = TestKernelStructScalarWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        scalar.encode(&input, output, copy, stride as u32, commands);
    });
    let mut copies = [Copy {
        source: 0,
        destination: 3,
    }; 16];
    copies[8] = copy;
    let constant = TestKernelStructConstantWebGPUKernel::new(&context, T::data_type()).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        constant.encode(&input, output, &copies, stride as u32, commands);
    });
}

#[uzu_test]
fn struct_buffer() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let size = 257;
    let ring = RingParams {
        ring_offset: 19,
        ring_length: 260,
    };
    let pairs = (0..size).flat_map(|i| [i * 3 + 1, i * 7 + 2]).collect::<Vec<u32>>();
    let input = create_buffer_with_data::<WebGPU, u32>(&context, &pairs);
    let initial = vec![u32::MAX; 2 * ring.ring_length as usize];
    let mut expected_pairs = initial.clone();
    let mut expected_indices = vec![u32::MAX; ring.ring_length as usize];
    for i in 0..size as usize {
        let dst = (i + ring.ring_offset as usize) % ring.ring_length as usize;
        expected_pairs[2 * dst..2 * dst + 2].copy_from_slice(&pairs[2 * i..2 * i + 2]);
        expected_indices[dst] = pairs[2 * i];
    }
    let mut indices = create_buffer_with_data::<WebGPU, u32>(&context, &vec![u32::MAX; ring.ring_length as usize]);
    let kernel = TestKernelStructBufferWebGPUKernel::new(&context).unwrap();
    check(&context, &initial, &expected_pairs, 0.0, |output, commands| {
        kernel.encode(&input, output, &mut indices, ring, size, commands);
    });
    assert_eq!(buffer_to_vec::<WebGPU, u32>(&context, &indices), expected_indices);
}

#[uzu_test]
fn thread_context() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let initial = vec![u32::MAX; 260];
    let mut expected = initial.clone();
    expected[..257].fill(0);
    let kernel = TestKernelThreadContextWebGPUKernel::new(&context).unwrap();
    check(&context, &initial, &expected, 0.0, |output, commands| {
        kernel.encode(output, 257, commands);
    });
}

macro_rules! float_tests {
    ($module:ident, $ty:ty, $($test:ident),+ $(,)?) => {
        mod $module {
            use super::*;

            $(
                #[uzu_test]
                fn $test() {
                    super::$test::<$ty>();
                }
            )+

            #[uzu_test]
            fn multispecialize_to_f32() {
                super::multispecialize::<$ty, f32>();
            }

            #[uzu_test]
            fn multispecialize_to_f16() {
                super::multispecialize::<$ty, half::f16>();
            }
        }
    };
}

float_tests!(
    f32,
    f32,
    dispatch_1d,
    dispatch_2d,
    specialize_and_optional,
    indirect,
    constant_array,
    groupshared,
    enums,
    struct_constants
);
float_tests!(
    f16,
    half::f16,
    dispatch_1d,
    dispatch_2d,
    specialize_and_optional,
    indirect,
    constant_array,
    groupshared,
    enums,
    struct_constants
);
