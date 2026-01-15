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
    type BackendBuffer;

    fn shape(&self) -> &[usize];

    fn num_elements(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    fn data_type(&self) -> DataType;

    fn label(&self) -> String;

    /// Returns a reference to the device buffer containing the array's data.
    fn buffer(&self) -> &[u8];

    /// Returns a mutable reference to the device buffer containing the array's data.
    fn buffer_mut(&mut self) -> &mut [u8];

    fn backend_buffer(&self) -> &Self::BackendBuffer;

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
}
