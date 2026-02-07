use std::rc::Rc;

use bytemuck::fill_zeroes;
use half::{bf16, f16};
use ndarray::{ArrayView, Dimension};
use num_traits::Zero;

use crate::{
    ArrayElement, DataType,
    array::{Array, size_for_shape},
    backends::common::{Backend, Context, NativeBuffer},
};

fn slice_index_to_array_index<const D: usize>(
    index: usize,
    shape: &[usize; D],
) -> [usize; D] {
    let mut indices = [0; D];
    let mut remaining_index = index;

    for (dim_index, dim_size) in shape.iter().enumerate().rev() {
        indices[dim_index] = remaining_index % dim_size;
        remaining_index = remaining_index / dim_size;
    }
    indices
}

pub trait DeviceContext {
    type Backend: Backend;

    /// Allocate a new array with the given shape and data type, but doesn't initialize it.
    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> Array<Self::Backend>;

    /// Allocate a new array with the given shape and data type.
    /// The result is guaranteed to be zero-initialized.
    fn array(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> Array<Self::Backend> {
        unsafe {
            let mut result = self.array_uninitialized(shape, data_type, label);
            fill_zeroes(result.as_bytes_mut());
            result
        }
    }

    /// Allocate a new array and populates it with the data copied from the given view.
    fn array_from_view<T: ArrayElement, D: Dimension>(
        &self,
        view: ArrayView<T, D>,
        label: String,
    ) -> Array<Self::Backend> {
        unsafe {
            let mut result =
                self.array_uninitialized(view.shape(), T::data_type(), label);
            let result_buffer = result.as_slice_mut::<T>();

            // If the view data is contiguous, copy it directly, otherwise .
            if let Some(slice) = view.as_slice_memory_order() {
                result_buffer.copy_from_slice(slice);
            } else {
                for (dst, src) in result_buffer.iter_mut().zip(view.iter()) {
                    *dst = *src;
                }
            }
            result
        }
    }

    /// Allocate a new array filled with values created by the function.
    fn array_from_shape_fn<
        const D: usize,
        T: ArrayElement,
        F: FnMut(&[usize; D]) -> T,
    >(
        &self,
        shape: &[usize; D],
        label: String,
        mut f: F,
    ) -> Array<Self::Backend> {
        unsafe {
            let mut result =
                self.array_uninitialized(shape, T::data_type(), label);
            for (slice_index, value) in
                result.as_slice_mut::<T>().iter_mut().enumerate()
            {
                *value = f(&slice_index_to_array_index(slice_index, shape));
            }
            result
        }
    }

    /// Allocate a new array filled with copies of elem.
    fn array_from_elem<T: ArrayElement>(
        &self,
        shape: &[usize],
        elem: T,
        label: String,
    ) -> Array<Self::Backend> {
        unsafe {
            let mut result =
                self.array_uninitialized(shape, T::data_type(), label);
            result.as_slice_mut::<T>().fill(elem);
            result
        }
    }

    /// Allocate a new array containing a single element.
    fn scalar<T: ArrayElement>(
        &self,
        value: T,
        label: String,
    ) -> Array<Self::Backend> {
        self.array_from_elem(&[], value, label)
    }

    /// Allocate a new array for an attention bias (also sometimes referred to as "attention mask").
    /// The result is -inf at indices where `should_be_neg_inf` is true, and zero otherwise.
    fn attention_bias<F>(
        &self,
        suffix_length: usize,
        prefix_length: usize,
        data_type: DataType,
        mut should_be_neg_inf: F,
    ) -> Array<Self::Backend>
    where
        F: FnMut(usize, usize) -> bool,
    {
        let shape = [suffix_length, suffix_length + prefix_length];
        let label = String::from("attention_bias");
        match data_type {
            DataType::F16 => {
                self.array_from_shape_fn(&shape, label, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        f16::NEG_INFINITY
                    } else {
                        f16::zero()
                    }
                })
            },
            DataType::BF16 => {
                self.array_from_shape_fn(&shape, label, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        bf16::NEG_INFINITY
                    } else {
                        bf16::zero()
                    }
                })
            },
            DataType::F32 => {
                self.array_from_shape_fn(&shape, label, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        f32::NEG_INFINITY
                    } else {
                        f32::zero()
                    }
                })
            },
            DataType::F64 => {
                self.array_from_shape_fn(&shape, label, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        f64::NEG_INFINITY
                    } else {
                        f64::zero()
                    }
                })
            },
            _ => {
                panic!(
                    "Attention bias can only be of a floating-point type, but requested {:?}",
                    data_type
                );
            },
        }
    }

    /// Copy data from the given ndarray view into an already allocated device array.
    /// The destination buffer must have enough capacity to hold the view element count.
    fn copy_from_view<T: ArrayElement, D: Dimension>(
        &self,
        dst: &mut Array<Self::Backend>,
        view: ArrayView<T, D>,
    ) {
        // Ensure data types match
        assert_eq!(dst.data_type(), T::data_type());

        let dst_slice = dst.as_slice_mut::<T>();

        if let Some(src_slice) = view.as_slice_memory_order() {
            assert!(src_slice.len() <= dst_slice.len());
            dst_slice[..src_slice.len()].copy_from_slice(src_slice);
        } else {
            // Fallback: iterate element-wise
            assert!(view.len() <= dst_slice.len());
            for (d, s) in dst_slice.iter_mut().zip(view.iter()) {
                *d = *s;
            }
        }
    }

    /// TODO: Create an DeviceBuffer entity that has slice access to data, but doesn't have shape or type
    fn fill_attention_bias<F>(
        &self,
        dst: &mut Array<Self::Backend>,
        suffix_length: usize,
        prefix_segment_length: usize,
        mut should_be_neg_inf: F,
    ) where
        F: FnMut(usize, usize) -> bool,
    {
        let total_elems =
            suffix_length * (suffix_length + prefix_segment_length);
        match dst.data_type() {
            DataType::F16 => {
                // TODO: Use lambda to avoid crimes
                let mut buf = Vec::<f16>::with_capacity(total_elems);
                for row in 0..suffix_length {
                    for col in 0..suffix_length + prefix_segment_length {
                        buf.push(if should_be_neg_inf(row, col) {
                            f16::NEG_INFINITY
                        } else {
                            f16::zero()
                        });
                    }
                }
                self.copy_from_view(dst, ndarray::ArrayView1::from(&buf));
            },
            DataType::BF16 => {
                let mut buf = Vec::<bf16>::with_capacity(total_elems);
                for row in 0..suffix_length {
                    for col in 0..suffix_length + prefix_segment_length {
                        buf.push(if should_be_neg_inf(row, col) {
                            bf16::NEG_INFINITY
                        } else {
                            bf16::zero()
                        });
                    }
                }
                self.copy_from_view(dst, ndarray::ArrayView1::from(&buf));
            },
            DataType::F32 => {
                let mut buf = Vec::<f32>::with_capacity(total_elems);
                for row in 0..suffix_length {
                    for col in 0..suffix_length + prefix_segment_length {
                        buf.push(if should_be_neg_inf(row, col) {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        });
                    }
                }
                self.copy_from_view(dst, ndarray::ArrayView1::from(&buf));
            },
            DataType::F64 => {
                let mut buf = Vec::<f64>::with_capacity(total_elems);
                for row in 0..suffix_length {
                    for col in 0..suffix_length + prefix_segment_length {
                        buf.push(if should_be_neg_inf(row, col) {
                            f64::NEG_INFINITY
                        } else {
                            0.0
                        });
                    }
                }
                self.copy_from_view(dst, ndarray::ArrayView1::from(&buf));
            },
            _ => panic!("Unsupported data type for attention bias fill"),
        }
    }
}

impl<C: Context> DeviceContext for C {
    type Backend = C::Backend;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> Array<C::Backend> {
        unsafe {
            let buffer_size_bytes = size_for_shape(shape, data_type);

            let buffer = self
                .create_buffer(buffer_size_bytes)
                .expect("Failed to create buffer");
            buffer.set_label(Some(&label));
            Array::from_parts(buffer, 0, shape, data_type)
        }
    }
}

impl<C: Context> DeviceContext for Rc<C> {
    type Backend = C::Backend;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> Array<C::Backend> {
        unsafe { self.as_ref().array_uninitialized(shape, data_type, label) }
    }
}
