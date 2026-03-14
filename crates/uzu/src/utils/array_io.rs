use half::{bf16, f16};

use crate::{DataType, array::Array, audio::AudioError, backends::common::Backend};

#[derive(Debug, thiserror::Error)]
pub enum ArrayIoError {
    #[error("array element count mismatch: expected {expected}, got {actual}")]
    ElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    #[error("unsupported dtype for {operation}: {data_type:?}")]
    UnsupportedDataType {
        operation: &'static str,
        data_type: DataType,
    },
}

impl From<ArrayIoError> for AudioError {
    fn from(value: ArrayIoError) -> Self {
        AudioError::Runtime(value.to_string())
    }
}

pub fn write_f32_slice_into_array<B: Backend>(
    array: &mut Array<B>,
    values: &[f32],
) -> Result<(), ArrayIoError> {
    if array.num_elements() != values.len() {
        return Err(ArrayIoError::ElementCountMismatch {
            expected: array.num_elements(),
            actual: values.len(),
        });
    }
    match array.data_type() {
        DataType::F32 => {
            array.as_slice_mut::<f32>().copy_from_slice(values);
            Ok(())
        },
        DataType::F16 => {
            for (dst, &src) in array.as_slice_mut::<f16>().iter_mut().zip(values.iter()) {
                *dst = f16::from_f32(src);
            }
            Ok(())
        },
        DataType::BF16 => {
            for (dst, &src) in array.as_slice_mut::<bf16>().iter_mut().zip(values.iter()) {
                *dst = bf16::from_f32(src);
            }
            Ok(())
        },
        data_type => Err(ArrayIoError::UnsupportedDataType {
            operation: "f32 array write",
            data_type,
        }),
    }
}

pub fn read_array_to_f32_vec<B: Backend>(array: &Array<B>) -> Result<Vec<f32>, ArrayIoError> {
    match array.data_type() {
        DataType::F32 => Ok(array.as_slice::<f32>().to_vec()),
        DataType::F16 => Ok(array.as_slice::<f16>().iter().copied().map(f32::from).collect()),
        DataType::BF16 => Ok(array.as_slice::<bf16>().iter().copied().map(f32::from).collect()),
        data_type => Err(ArrayIoError::UnsupportedDataType {
            operation: "f32 array read",
            data_type,
        }),
    }
}

pub fn write_i32_slice_into_array<B: Backend>(
    array: &mut Array<B>,
    values: &[i32],
) -> Result<(), ArrayIoError> {
    if array.num_elements() != values.len() {
        return Err(ArrayIoError::ElementCountMismatch {
            expected: array.num_elements(),
            actual: values.len(),
        });
    }
    match array.data_type() {
        DataType::I32 => {
            array.as_slice_mut::<i32>().copy_from_slice(values);
            Ok(())
        },
        data_type => Err(ArrayIoError::UnsupportedDataType {
            operation: "i32 array write",
            data_type,
        }),
    }
}
