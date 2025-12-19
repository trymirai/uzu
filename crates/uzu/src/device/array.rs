use std::ops::Range;

use bytemuck;
use ndarray::{ArrayView, ArrayViewMut, IxDyn};
use thiserror::Error;

use crate::{ArrayElement, DataType};

#[derive(Error, Debug)]
pub enum ArrayConversionError {
    #[error("Invalid data type, expected {expected:?}, actual {actual:?}")]
    InvalidDataType {
        expected: DataType,
        actual: DataType,
    },
    #[error("Array of shape {0:?} is not a scalar")]
    NotAScalar(Box<[usize]>),
}

/// Calculate the number of bytes required to store an array with a given shape and DataType.
/// Takes into account alignment of packed sub-byte data types.
pub fn array_size_in_bytes(
    shape: &[usize],
    data_type: DataType,
) -> usize {
    if shape.is_empty() {
        return data_type.size_in_bytes();
    }
    let last_dim = *shape.last().unwrap();
    let bits_per_row = last_dim * data_type.size_in_bits();
    let padded_bytes_per_row = (bits_per_row + 7) / 8;

    let num_rows: usize = shape.iter().rev().skip(1).product();

    num_rows * padded_bytes_per_row
}

// Backend-agnostic n-dimensional array type.
pub trait Array {
    fn shape(&self) -> &[usize];

    fn num_elements(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    fn data_type(&self) -> DataType;

    /// Returns a reference to the device buffer containing the array's data.
    fn buffer(&self) -> &[u8];

    /// Returns a mutable reference to the device buffer containing the array's data.
    fn buffer_mut(&mut self) -> &mut [u8];

    /// Returns the size of the array in memory.
    fn size_in_bytes(&self) -> usize {
        array_size_in_bytes(self.shape(), self.data_type())
    }

    /// Returns true if the array holds a single scalar value.
    /// Equivalent to `self.shape().is_empty()`.
    fn is_scalar(&self) -> bool {
        self.shape().is_empty()
    }

    /// Returns a reference to the scalar value contained in the array.
    /// Returns an error if the array is not a scalar or if the requested type does not match the array's data type.
    fn item<T: ArrayElement>(&self) -> Result<&T, ArrayConversionError> {
        if !self.is_scalar() {
            Err(ArrayConversionError::NotAScalar(self.shape().into()))
        } else {
            let casted_buffer: &[T] = bytemuck::cast_slice(self.buffer());
            assert_eq!(casted_buffer.len(), 1);
            Ok(&casted_buffer.first().unwrap())
        }
    }

    /// Returns a mutable reference to the scalar value contained in the array.
    /// Returns an error if the array is not a scalar or if the requested type does not match the array's data type.
    fn item_mut<T: ArrayElement>(
        &mut self
    ) -> Result<&mut T, ArrayConversionError> {
        if !self.is_scalar() {
            Err(ArrayConversionError::NotAScalar(self.shape().into()))
        } else {
            let casted_buffer: &mut [T] =
                bytemuck::cast_slice_mut(self.buffer_mut());
            assert_eq!(casted_buffer.len(), 1);
            let first = casted_buffer.first_mut();
            Ok(first.unwrap())
        }
    }

    /// Returns an typed slice reffering to the underlying buffer.
    /// Returns an error if the requested type does not match the array's data type.
    fn as_slice<T: ArrayElement>(&self) -> Result<&[T], ArrayConversionError> {
        if T::data_type() != self.data_type() {
            return Err(ArrayConversionError::InvalidDataType {
                expected: T::data_type(),
                actual: self.data_type(),
            });
        }
        Ok(bytemuck::cast_slice(self.buffer()))
    }

    /// Returns an mutable typed slice reffering to the underlying buffer.
    /// Returns an error if the requested type does not match the array's data type.
    fn as_slice_mut<T: ArrayElement>(
        &mut self
    ) -> Result<&mut [T], ArrayConversionError> {
        if T::data_type() != self.data_type() {
            return Err(ArrayConversionError::InvalidDataType {
                expected: T::data_type(),
                actual: self.data_type(),
            });
        }
        Ok(bytemuck::cast_slice_mut(self.buffer_mut()))
    }

    /// Returns an ndarray::ArrayView of the underlying buffer.
    /// Returns an error if the requested type does not match the array's data type.
    fn as_view<T: ArrayElement>(
        &self
    ) -> Result<ArrayView<'_, T, IxDyn>, ArrayConversionError> {
        Ok(ArrayView::from_shape(self.shape(), self.as_slice()?).unwrap())
    }

    /// Returns an ndarray::ArrayViewMut of the underlying buffer.
    /// Returns an error if the requested type does not match the array's data type.
    fn as_view_mut<T: ArrayElement>(
        &mut self
    ) -> Result<ArrayViewMut<'_, T, IxDyn>, ArrayConversionError> {
        Ok(ArrayViewMut::from_shape(
            self.shape().to_owned(),
            self.as_slice_mut()?,
        )
        .unwrap())
    }

    /// Create a new array sharing the same underlying buffer but with a different shape.
    /// The total size in bytes must be less than or equal to the underlying buffer size.
    fn reshape(
        &self,
        shape: &[usize],
    ) -> Self
    where
        Self: Sized;

    fn copy_from(
        &mut self,
        other: &Self,
    ) {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        assert_eq!(self.data_type(), other.data_type(), "Data type mismatch");
        self.buffer_mut().copy_from_slice(other.buffer());
    }

    fn copy_slice(
        &mut self,
        source: &Self,
        axis: usize,
        src_range: Range<usize>,
        dst_offset: usize,
    ) {
        assert_eq!(self.shape().len(), source.shape().len(), "Rank mismatch");
        assert!(axis < self.shape().len(), "Axis out of bounds");
        for (i, (a, b)) in self.shape().iter().zip(source.shape()).enumerate() {
            if i != axis {
                assert_eq!(a, b, "Shapes must match on all non-sliced axes");
            }
        }
        assert_eq!(
            self.data_type(),
            source.data_type(),
            "Arrays must have the same data type"
        );

        let elem_size = self.data_type().size_in_bytes();

        // The number of contiguous elements to copy for each slice operation
        let block_size_elems =
            self.shape().iter().skip(axis + 1).product::<usize>();
        let block_size_bytes = block_size_elems * elem_size;

        // The total number of blocks to copy
        let num_blocks: usize = self.shape().iter().take(axis).product();

        // Strides between the start of each block
        let src_stride_bytes = source.shape()[axis] * block_size_bytes;
        let dst_stride_bytes = self.shape()[axis] * block_size_bytes;

        let rows_to_copy = src_range.end - src_range.start;
        assert!(dst_offset + rows_to_copy <= self.shape()[axis]);
        assert!(src_range.end <= source.shape()[axis]);

        let src_buf = source.buffer();
        let dst_buf = self.buffer_mut();
        let copy_bytes = rows_to_copy * block_size_bytes;

        for i in 0..num_blocks {
            let src_block_start = i * src_stride_bytes;
            let dst_block_start = i * dst_stride_bytes;

            let src_start =
                src_block_start + src_range.start * block_size_bytes;
            let dst_start = dst_block_start + dst_offset * block_size_bytes;

            let src_slice = &src_buf[src_start..src_start + copy_bytes];
            let dst_slice = &mut dst_buf[dst_start..dst_start + copy_bytes];
            dst_slice.copy_from_slice(src_slice);
        }
    }
}
