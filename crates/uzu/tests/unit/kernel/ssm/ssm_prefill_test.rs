use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{SSDPrefillKernel, SSDPrefillSequentialKernel},
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    x: Box<[T]>,
    dt: Box<[T]>,
    b: Box<[T]>,
    c: Box<[T]>,
    d: Box<[T]>,
    z: Box<[T]>,
    state: Box<[T]>,
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
    x_strides: [usize; 3],
    dt_strides: [usize; 2],
    cb_strides: [usize; 3],
    state_strides: [usize; 3],
}

struct Output<T: ArrayElement + Float> {
    y: Vec<T>,
    state: Vec<T>,
}

enum KernelType {
    Prefill,
    Sequential,
}

fn get_input<T: ArrayElement + Float>(
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
) -> Input<T> {
    let safe_group = group_size.max(1) as usize;
    let group_count = num_heads / safe_group;
    let total_x = suffix_len * num_heads * head_dim;
    let total_dt = suffix_len * num_heads;
    let total_cb = suffix_len * group_count * state_dim;
    let total_state = num_heads * head_dim * state_dim;

    let x: Vec<T> = (0..total_x).map(|i| T::from(((i % 17) as f64) * 0.01 - 0.05).unwrap()).collect();
    let dt: Vec<T> = (0..total_dt).map(|i| T::from(((i % 13) as f64) * 0.2 - 1.5).unwrap()).collect();
    let b: Vec<T> = (0..total_cb).map(|i| T::from(((i % 11) as f64) * 0.02 - 0.05).unwrap()).collect();
    let c: Vec<T> = (0..total_cb).map(|i| T::from(((i % 19) as f64) * 0.01 - 0.02).unwrap()).collect();
    let d: Vec<T> = (0..num_heads).map(|i| T::from(((i % 3) as f64) * 0.05 - 0.05).unwrap()).collect();
    let z: Vec<T> = (0..total_x).map(|i| T::from(((i % 23) as f64) * 0.02 - 0.1).unwrap()).collect();
    let state: Vec<T> = (0..total_state).map(|i| T::from(((i % 29) as f64) * 0.03 - 0.4).unwrap()).collect();

    let x_strides = [num_heads * head_dim, head_dim, 1];
    let dt_strides = [num_heads, 1];
    let cb_strides = [group_count * state_dim, state_dim, 1];
    let state_strides = [head_dim * state_dim, state_dim, 1];

    Input {
        x: x.into_boxed_slice(),
        dt: dt.into_boxed_slice(),
        b: b.into_boxed_slice(),
        c: c.into_boxed_slice(),
        d: d.into_boxed_slice(),
        z: z.into_boxed_slice(),
        state: state.into_boxed_slice(),
        suffix_len,
        num_heads,
        head_dim,
        state_dim,
        group_size,
        x_strides,
        dt_strides,
        cb_strides,
        state_strides,
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(
    input: &Input<T>,
    kernel_type: &KernelType,
) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let total_x = input.suffix_len * input.num_heads * input.head_dim;

    let x_array = context.create_array_from(&[input.x.len()], &input.x, "x");
    let dt_array = context.create_array_from(&[input.dt.len()], &input.dt, "dt");
    let b_array = context.create_array_from(&[input.b.len()], &input.b, "b");
    let c_array = context.create_array_from(&[input.c.len()], &input.c, "c");
    let d_array = context.create_array_from(&[input.d.len()], &input.d, "d");
    let z_array = context.create_array_from(&[input.z.len()], &input.z, "z");
    let state_array = context.create_array_from(&[input.state.len()], &input.state, "state");
    let y_array = context.create_array_uninitialized(&[total_x], T::data_type(), "y");

    let x_strides: Vec<u32> = input.x_strides.iter().map(|&s| s as u32).collect();
    let dt_strides: Vec<u32> = input.dt_strides.iter().map(|&s| s as u32).collect();
    let cb_strides: Vec<u32> = input.cb_strides.iter().map(|&s| s as u32).collect();
    let state_strides: Vec<u32> = input.state_strides.iter().map(|&s| s as u32).collect();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    match kernel_type {
        KernelType::Prefill => {
            let kernel = <<B as Backend>::Kernels as Kernels>::SSDPrefillKernel::new(&context, T::data_type())
                .expect("Failed to create SSDPrefillKernel");
            kernel.encode(
                x_array.buffer().borrow().deref(),
                dt_array.buffer().borrow().deref(),
                b_array.buffer().borrow().deref(),
                c_array.buffer().borrow().deref(),
                d_array.buffer().borrow().deref(),
                z_array.buffer().borrow().deref(),
                state_array.buffer().borrow_mut().deref_mut(),
                y_array.buffer().borrow_mut().deref_mut(),
                input.suffix_len as u32,
                input.group_size as u32,
                input.state_dim as u32,
                &x_strides,
                &dt_strides,
                &cb_strides,
                &state_strides,
                input.num_heads as u32,
                input.head_dim as u32,
                &mut encoder,
            );
        },
        KernelType::Sequential => {
            let kernel =
                <<B as Backend>::Kernels as Kernels>::SSDPrefillSequentialKernel::new(&context, T::data_type())
                    .expect("Failed to create SSDPrefillSequentialKernel");
            kernel.encode(
                x_array.buffer().borrow().deref(),
                dt_array.buffer().borrow().deref(),
                b_array.buffer().borrow().deref(),
                c_array.buffer().borrow().deref(),
                d_array.buffer().borrow().deref(),
                z_array.buffer().borrow().deref(),
                state_array.buffer().borrow_mut().deref_mut(),
                y_array.buffer().borrow_mut().deref_mut(),
                input.suffix_len as u32,
                input.group_size as u32,
                input.state_dim as u32,
                &x_strides,
                &dt_strides,
                &cb_strides,
                &state_strides,
                input.num_heads as u32,
                input.head_dim as u32,
                &mut encoder,
            );
        },
    };
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        y: y_array.as_slice().to_vec(),
        state: state_array.as_slice().to_vec(),
    }
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &Output<T>,
    kernel_type: &KernelType,
    label: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2f32
    } else {
        5e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input, kernel_type);
        let backend_name = std::any::type_name::<B>();
        let type_name = std::any::type_name::<T>();

        assert_eq_float::<T>(
            &expected.y,
            &output.y,
            eps,
            &format!("SSDPrefillSequential y {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.state,
            &output.state,
            eps,
            &format!("SSDPrefillSequential state {backend_name} {label} (type={type_name})"),
        );
    });
}

// --- test shapes ---

fn test_shape(
    kernel_type: &KernelType,
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
    label: &str,
) {
    fn run<T: ArrayElement + Float + Debug + Display>(
        kernel_type: &KernelType,
        suffix_len: usize,
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
        group_size: u32,
        label: &str,
    ) {
        let input = get_input::<T>(suffix_len, num_heads, head_dim, state_dim, group_size);
        let expected = get_output::<Cpu, T>(&input, &KernelType::Sequential);
        test_internal(&input, &expected, kernel_type, label);
    }
    run::<f32>(kernel_type, suffix_len, num_heads, head_dim, state_dim, group_size, label);
    run::<f16>(kernel_type, suffix_len, num_heads, head_dim, state_dim, group_size, label);
    run::<bf16>(kernel_type, suffix_len, num_heads, head_dim, state_dim, group_size, label);
}

// --- Sequential ---
#[test]
fn test_sequential_basic() {
    test_shape(&KernelType::Sequential, 512, 32, 64, 64, 1, "sequential_basic");
}

#[test]
fn test_sequential_small() {
    test_shape(&KernelType::Sequential, 4, 4, 4, 8, 1, "sequential_small");
}

#[test]
fn test_sequential_minimal() {
    test_shape(&KernelType::Sequential, 1, 1, 1, 1, 1, "sequential_minimal");
}

#[test]
fn test_sequential_multi_group() {
    test_shape(&KernelType::Sequential, 8, 8, 4, 16, 4, "sequential_multi_group");
}

#[test]
fn test_sequential_group_per_head() {
    test_shape(&KernelType::Sequential, 8, 4, 4, 8, 1, "sequential_group_per_head");
}

// --- Prefill ---
#[test]
fn test_prefill_basic() {
    test_shape(&KernelType::Prefill, 512, 32, 64, 64, 1, "prefill_basic");
}

#[test]
fn test_prefill_small() {
    test_shape(&KernelType::Prefill, 4, 4, 4, 8, 1, "prefill_small");
}

#[test]
fn test_prefill_minimal() {
    test_shape(&KernelType::Prefill, 1, 1, 1, 1, 1, "prefill_minimal");
}

#[test]
fn test_prefill_multi_group() {
    test_shape(&KernelType::Prefill, 8, 8, 4, 16, 4, "prefill_multi_group");
}

#[test]
fn test_prefill_group_per_head() {
    test_shape(&KernelType::Prefill, 8, 4, 4, 8, 1, "prefill_group_per_head");
}
