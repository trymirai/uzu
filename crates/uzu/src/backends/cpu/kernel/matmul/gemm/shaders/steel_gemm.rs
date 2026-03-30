use num_traits::Float;

use crate::{
    ArrayElement,
    utils::pointers::{SendPtr, SendPtrMut},
};

#[inline(always)]
unsafe fn matmul_gemm_compute_row<T: ArrayElement + Float>(
    a: SendPtr<T>,
    b: SendPtr<T>,
    ab_scale: f32,
    d: SendPtrMut<T>,
    row: usize,
    n: usize,
    k: usize,
    leading_dimension_a: usize,
    leading_dimension_b: usize,
    leading_dimension_d: usize,
    is_accumulate: bool,
) {
    unsafe {
        for col in 0..n {
            let mut acc: f32 = 0.0;
            for i in 0..k {
                let a_val = *a.0.add(row * leading_dimension_a + i);
                let b_val = *b.0.add(col * leading_dimension_b + i);
                acc = acc + a_val.to_f32().unwrap() * b_val.to_f32().unwrap();
            }
            let d_element = d.0.add(row * leading_dimension_d + col);
            acc *= ab_scale;
            if is_accumulate {
                acc += (*d_element).to_f32().unwrap();
            }
            *d_element = T::from(acc).unwrap();
        }
    }
}

pub fn matmul_gemm_impl<T: ArrayElement + Float>(
    a: *const T,
    b: *const T,
    ab_scale: f32,
    d: *mut T,
    m: usize,
    n: usize,
    k: usize,
    leading_dimension_a: usize,
    leading_dimension_b: usize,
    leading_dimension_d: usize,
    is_accumulate: bool,
) {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let a = SendPtr(a);
    let b = SendPtr(b);
    let d = SendPtrMut(d);

    unsafe {
        std::thread::scope(|s| {
            let rows_per_thread = m / num_threads;
            let rows_per_thread_remainder = m % num_threads;
            let mut start = 0;

            for t in 0..num_threads {
                let chunk_add = if t < rows_per_thread_remainder {
                    1
                } else {
                    0
                };
                let chunk = rows_per_thread + chunk_add;
                let end = start + chunk;

                s.spawn(move || unsafe {
                    for row in start..end {
                        matmul_gemm_compute_row(
                            a,
                            b,
                            ab_scale,
                            d,
                            row,
                            n,
                            k,
                            leading_dimension_a,
                            leading_dimension_b,
                            leading_dimension_d,
                            is_accumulate,
                        );
                    }
                });

                start = end;
            }
        });
    }
}
