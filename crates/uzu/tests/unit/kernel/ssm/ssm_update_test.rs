use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::SSDUpdateKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    x: Box<[T]>,
    dt: Box<[T]>,
    b: Box<[T]>,
    c: Box<[T]>,
    d: Box<[T]>,
    z: Box<[T]>,
    state: Box<[T]>,
    bsz: u32,
    h: u32,
    dh: u32,
    g: u32,
    n: u32,
    state_in_place: bool,
}

struct Output<T: ArrayElement + Float> {
    y: Vec<T>,
    next_state: Vec<T>,
}

fn get_input<T: ArrayElement + Float>(
    bsz: u32,
    h: u32,
    dh: u32,
    g: u32,
    n: u32,
    state_in_place: bool,
) -> Input<T> {
    let bsz_u = bsz as usize;
    let h_u = h as usize;
    let dh_u = dh as usize;
    let g_u = g as usize;
    let n_u = n as usize;

    let x: Vec<T> = (0..bsz_u * h_u * dh_u).map(|i| T::from(((i % 7) as f64) * 0.1 - 0.2).unwrap()).collect();
    let z: Vec<T> = (0..bsz_u * h_u * dh_u).map(|i| T::from(((i % 5) as f64) * 0.1 - 0.1).unwrap()).collect();
    let dt: Vec<T> = (0..bsz_u * h_u).map(|i| T::from(((i % 5) as f64) * 0.3 - 1.0).unwrap()).collect();
    let b: Vec<T> = (0..bsz_u * g_u * n_u).map(|i| T::from(((i % 11) as f64) * 0.02 - 0.05).unwrap()).collect();
    let c: Vec<T> = (0..bsz_u * g_u * n_u).map(|i| T::from(((i % 13) as f64) * 0.015).unwrap()).collect();
    let d: Vec<T> = (0..h_u).map(|i| T::from(((i % 3) as f64) * 0.05).unwrap()).collect();
    let state: Vec<T> =
        (0..bsz_u * h_u * dh_u * n_u).map(|i| T::from(((i % 23) as f64) * 0.01 - 0.05).unwrap()).collect();

    Input {
        x: x.into_boxed_slice(),
        dt: dt.into_boxed_slice(),
        b: b.into_boxed_slice(),
        c: c.into_boxed_slice(),
        d: d.into_boxed_slice(),
        z: z.into_boxed_slice(),
        state: state.into_boxed_slice(),
        bsz,
        h,
        dh,
        g,
        n,
        state_in_place,
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::SSDUpdateKernel::new(&context, T::data_type(), input.state_in_place)
            .expect("Failed to create SSDUpdateKernel");

    let bsz = input.bsz as usize;
    let h = input.h as usize;
    let dh = input.dh as usize;
    let g = input.g as usize;
    let n = input.n as usize;

    let x_strides = [(h * dh) as u32, dh as u32, 1u32];
    let dt_strides = [h as u32, 1u32];
    let cb_strides = [(g * n) as u32, n as u32, 1u32];
    let state_strides = [(h * dh * n) as u32, (dh * n) as u32, n as u32, 1u32];

    let y_size = bsz * h * dh;
    let ns_size = bsz * h * dh * n;

    let x_array = context.create_array_from(&[input.x.len()], &input.x, "x");
    let dt_array = context.create_array_from(&[input.dt.len()], &input.dt, "dt");
    let b_array = context.create_array_from(&[input.b.len()], &input.b, "b");
    let c_array = context.create_array_from(&[input.c.len()], &input.c, "c");
    let d_array = context.create_array_from(&[input.d.len()], &input.d, "d");
    let z_array = context.create_array_from(&[input.z.len()], &input.z, "z");

    let y_array = context.create_array_uninitialized(&[y_size], T::data_type(), "y");

    if input.state_in_place {
        let ns_array = context.create_array_from(&[ns_size], &input.state, "next_state");

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        kernel.encode(
            x_array.buffer().borrow().deref(),
            dt_array.buffer().borrow().deref(),
            b_array.buffer().borrow().deref(),
            c_array.buffer().borrow().deref(),
            d_array.buffer().borrow().deref(),
            z_array.buffer().borrow().deref(),
            None::<&<B as Backend>::Buffer>,
            y_array.buffer().borrow_mut().deref_mut(),
            ns_array.buffer().borrow_mut().deref_mut(),
            (h / g) as u32,
            input.n,
            x_strides.as_slice(),
            dt_strides.as_slice(),
            cb_strides.as_slice(),
            state_strides.as_slice(),
            input.bsz,
            input.h,
            input.dh,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

        Output {
            y: y_array.as_slice().to_vec(),
            next_state: ns_array.as_slice().to_vec(),
        }
    } else {
        let state_array = context.create_array_from(&[ns_size], &input.state, "state");
        let ns_array = context.create_array_uninitialized(&[ns_size], T::data_type(), "next_state");

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        kernel.encode(
            x_array.buffer().borrow().deref(),
            dt_array.buffer().borrow().deref(),
            b_array.buffer().borrow().deref(),
            c_array.buffer().borrow().deref(),
            d_array.buffer().borrow().deref(),
            z_array.buffer().borrow().deref(),
            Some(state_array.buffer().borrow().deref()),
            y_array.buffer().borrow_mut().deref_mut(),
            ns_array.buffer().borrow_mut().deref_mut(),
            (h / g) as u32,
            input.n,
            x_strides.as_slice(),
            dt_strides.as_slice(),
            cb_strides.as_slice(),
            state_strides.as_slice(),
            input.bsz,
            input.h,
            input.dh,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

        Output {
            y: y_array.as_slice().to_vec(),
            next_state: ns_array.as_slice().to_vec(),
        }
    }
}

fn get_test_data<T: ArrayElement + Float>(
    bsz: u32,
    h: u32,
    dh: u32,
    g: u32,
    n: u32,
    state_in_place: bool,
) -> (Input<T>, Output<T>) {
    let input = get_input::<T>(bsz, h, dh, g, n, state_in_place);
    let expected = get_output::<Cpu, T>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &Output<T>,
    label: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let backend_name = std::any::type_name::<B>();
        let type_name = std::any::type_name::<T>();

        assert_eq_float::<T>(
            &expected.y,
            &output.y,
            eps,
            &format!("SSDUpdate y {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.next_state,
            &output.next_state,
            eps,
            &format!("SSDUpdate next_state {backend_name} {label} (type={type_name})"),
        );
    });
}

// bsz=1, h=4, dh=3, g=2, n=8 (matches existing Metal test)
fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 4, 3, 2, 8, false);
    test_internal(&input, &expected, "basic");
}

// bsz=2 multi-batch
fn test_multi_batch<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(2, 4, 3, 2, 8, false);
    test_internal(&input, &expected, "multi_batch");
}

// state_in_place=true
fn test_state_in_place<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 4, 3, 2, 8, true);
    test_internal(&input, &expected, "state_in_place");
}

// Larger dimensions closer to real usage
fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 8, 4, 4, 16, false);
    test_internal(&input, &expected, "large");
}

// Minimal: single head, single group, dh=1, n=1
fn test_minimal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 1, 1, 1, 1, false);
    test_internal(&input, &expected, "minimal");
}

// g == h (each head is its own group)
fn test_group_per_head<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 4, 2, 4, 8, false);
    test_internal(&input, &expected, "group_per_head");
}

// f32
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_multi_batch_f32() {
    test_multi_batch::<f32>();
}

#[uzu_test]
fn test_state_in_place_f32() {
    test_state_in_place::<f32>();
}

#[uzu_test]
fn test_large_f32() {
    test_large::<f32>();
}

#[uzu_test]
fn test_minimal_f32() {
    test_minimal::<f32>();
}

#[uzu_test]
fn test_group_per_head_f32() {
    test_group_per_head::<f32>();
}

// f16
#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_multi_batch_f16() {
    test_multi_batch::<f16>();
}

#[uzu_test]
fn test_state_in_place_f16() {
    test_state_in_place::<f16>();
}

#[uzu_test]
fn test_large_f16() {
    test_large::<f16>();
}

// bf16
#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[uzu_test]
fn test_multi_batch_bf16() {
    test_multi_batch::<bf16>();
}

#[uzu_test]
fn test_state_in_place_bf16() {
    test_state_in_place::<bf16>();
}

#[uzu_test]
fn test_large_bf16() {
    test_large::<bf16>();
}
