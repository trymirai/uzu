use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::SSDPrefillKernel},
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

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

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let total_x = input.suffix_len * input.num_heads * input.head_dim;

    let x = alloc_allocation_with_data::<B, T>(&context, &input.x);
    let dt = alloc_allocation_with_data::<B, T>(&context, &input.dt);
    let b = alloc_allocation_with_data::<B, T>(&context, &input.b);
    let c = alloc_allocation_with_data::<B, T>(&context, &input.c);
    let d = alloc_allocation_with_data::<B, T>(&context, &input.d);
    let z = alloc_allocation_with_data::<B, T>(&context, &input.z);
    let mut state = alloc_allocation_with_data::<B, T>(&context, &input.state);
    let mut y = alloc_allocation::<B, T>(&context, total_x);

    let x_strides: [u32; 3] = input.x_strides.map(|s| s as u32);
    let dt_strides: [u32; 2] = input.dt_strides.map(|s| s as u32);
    let cb_strides: [u32; 3] = input.cb_strides.map(|s| s as u32);
    let state_strides: [u32; 3] = input.state_strides.map(|s| s as u32);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let kernel = <<B as Backend>::Kernels as Kernels>::SSDPrefillKernel::new(&context, T::data_type())
        .expect("Failed to create SSDPrefillKernel");
    kernel.encode(
        &x,
        &dt,
        &b,
        &c,
        &d,
        &z,
        &mut state,
        &mut y,
        input.suffix_len as u32,
        input.group_size,
        input.state_dim as u32,
        &x_strides,
        &dt_strides,
        &cb_strides,
        &state_strides,
        input.num_heads as u32,
        input.head_dim as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        y: allocation_to_vec(&y),
        state: allocation_to_vec(&state),
    }
}

fn reference_output<T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let safe_group = input.group_size.max(1) as usize;
    let mut state = input.state.to_vec();
    let mut y = vec![T::zero(); input.suffix_len * input.num_heads * input.head_dim];

    for h in 0..input.num_heads {
        let group_idx = h / safe_group;
        for dh in 0..input.head_dim {
            let state_base = h * input.state_strides[0] + dh * input.state_strides[1];
            for token in 0..input.suffix_len {
                let x_idx = token * input.x_strides[0] + h * input.x_strides[1] + dh * input.x_strides[2];
                let dt_idx = token * input.dt_strides[0] + h * input.dt_strides[1];
                let cb_base = token * input.cb_strides[0] + group_idx * input.cb_strides[1];

                let x_val = input.x[x_idx];
                let dt_val = ActivationType::SOFTPLUS.activate(input.dt[dt_idx]);
                let decay_val = (-dt_val).exp();

                let mut acc = input.d[h] * x_val;
                for s in 0..input.state_dim {
                    let state_idx = state_base + s * input.state_strides[2];
                    let cb_idx = cb_base + s * input.cb_strides[2];
                    let new_state = decay_val * state[state_idx] + x_val * input.b[cb_idx];
                    state[state_idx] = new_state;
                    acc = acc + new_state * input.c[cb_idx];
                }

                let gate = ActivationType::SILU.activate(input.z[x_idx]);
                y[x_idx] = acc * gate;
            }
        }
    }

    Output {
        y,
        state,
    }
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &Output<T>,
    label: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2f32
    } else {
        5e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let backend_name = std::any::type_name::<B>();
        let type_name = std::any::type_name::<T>();

        assert_eq_float::<T>(
            &expected.y,
            &output.y,
            eps,
            &format!("SSDPrefill y {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.state,
            &output.state,
            eps,
            &format!("SSDPrefill state {backend_name} {label} (type={type_name})"),
        );
    });
}

// --- test shapes ---

fn test_shape(
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
    label: &str,
) {
    fn run<T: ArrayElement + Float + Debug + Display>(
        suffix_len: usize,
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
        group_size: u32,
        label: &str,
    ) {
        let input = get_input::<T>(suffix_len, num_heads, head_dim, state_dim, group_size);
        let expected = reference_output(&input);
        test_internal(&input, &expected, label);
    }
    run::<f32>(suffix_len, num_heads, head_dim, state_dim, group_size, label);
    run::<f16>(suffix_len, num_heads, head_dim, state_dim, group_size, label);
    run::<bf16>(suffix_len, num_heads, head_dim, state_dim, group_size, label);
}

// --- Prefill ---
#[uzu_test]
fn test_prefill_basic() {
    test_shape(512, 32, 64, 64, 1, "prefill_basic");
}

#[uzu_test]
fn test_prefill_small() {
    test_shape(4, 4, 4, 8, 1, "prefill_small");
}

#[uzu_test]
fn test_prefill_minimal() {
    test_shape(1, 1, 1, 1, 1, "prefill_minimal");
}

#[uzu_test]
fn test_prefill_multi_group() {
    test_shape(8, 8, 4, 16, 4, "prefill_multi_group");
}

#[uzu_test]
fn test_prefill_group_per_head() {
    test_shape(8, 4, 4, 8, 1, "prefill_group_per_head");
}
