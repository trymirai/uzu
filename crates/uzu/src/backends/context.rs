use bytemuck::fill_zeroes;
use ndarray::{ArrayView, Dimension};

use super::Backend;
use crate::{Array, ArrayElement, DataType};

pub trait Context<B: Backend + ?Sized>
where
    Self: Sized,
{
    fn default() -> Option<Self>;

    /// Allocate a new array with the given shape and data type, but doesn't initialize it.
    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> B::Array;

    /// Allocate a new array with the given shape and data type.
    /// The result is guaranteed to be zero-initialized.
    fn array(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> B::Array {
        unsafe {
            let mut result = self.array_uninitialized(shape, data_type);
            fill_zeroes(result.buffer_mut());
            result
        }
    }

    /// Allocate a new array and populates it with the data copied from the given view.
    fn array_from_view<T: ArrayElement, D: Dimension>(
        &self,
        view: ArrayView<T, D>,
    ) -> B::Array {
        unsafe {
            let mut result =
                self.array_uninitialized(view.shape(), T::data_type());
            let result_buffer = result.as_slice_mut().unwrap();

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
        mut f: F,
    ) -> B::Array {
        unsafe {
            let mut result = self.array_uninitialized(shape, T::data_type());
            for (slice_index, value) in
                result.as_slice_mut().unwrap().iter_mut().enumerate()
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
    ) -> B::Array {
        unsafe {
            let mut result = self.array_uninitialized(shape, T::data_type());
            result.as_slice_mut().unwrap().fill(elem);
            result
        }
    }

    /// Allocate a new array containing a single element.
    fn scalar<T: ArrayElement>(
        &self,
        value: T,
    ) -> B::Array {
        self.array_from_elem(&[], value)
    }
}

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
