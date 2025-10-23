use metal::{Device as MTLDevice, MTLResourceOptions};
use xgrammar::{DLDataType, DLDataTypeCode, DLDeviceType, DLTensor};

use crate::{DataType, backends::metal::MetalArray};

/// Error type for DLTensor conversion operations
#[derive(Debug, thiserror::Error)]
pub enum DLTensorError {
    #[error("Unsupported device type: expected Metal (8), got {0}")]
    UnsupportedDevice(i32),

    #[error("Unsupported data type: code={code}, bits={bits}, lanes={lanes}")]
    UnsupportedDataType {
        code: u8,
        bits: u8,
        lanes: u16,
    },

    #[error("Non-contiguous tensors are not supported (strides must be NULL)")]
    NonContiguous,

    #[error("Invalid shape: ndim={ndim} but shape pointer is NULL")]
    InvalidShape {
        ndim: i32,
    },

    #[error("Vectorized data types (lanes > 1) are not supported")]
    VectorizedType,

    #[error("Data pointer is NULL")]
    NullDataPointer,
}

/// Extension trait for converting DLTensor to uzu types
pub trait DLTensorExt {
    /// Convert DLTensor to MetalArray
    ///
    /// # Safety
    /// - The DLTensor must have valid pointers for the lifetime of the operation
    /// - The data pointer must point to a valid Metal buffer
    /// - The tensor must be contiguous (strides == NULL)
    /// - The tensor must be on a Metal device
    unsafe fn to_metal_array(
        &self,
        device: &MTLDevice,
    ) -> Result<MetalArray, DLTensorError>;

    /// Convert DLDataType to uzu DataType
    fn dl_dtype_to_data_type(
        dtype: &DLDataType
    ) -> Result<DataType, DLTensorError>;
}

impl DLTensorExt for DLTensor {
    unsafe fn to_metal_array(
        &self,
        device: &MTLDevice,
    ) -> Result<MetalArray, DLTensorError> {
        // Validate device type
        let device_type_value = match self.device.device_type {
            DLDeviceType::kDLMetal => 8,
            DLDeviceType::kDLCPU => 1,
            DLDeviceType::kDLCUDA => 2,
            _ => -1,
        };

        if device_type_value != 8 {
            return Err(DLTensorError::UnsupportedDevice(device_type_value));
        }

        // Validate tensor is contiguous
        if !self.strides.is_null() {
            return Err(DLTensorError::NonContiguous);
        }

        // Validate data pointer
        if self.data.is_null() {
            return Err(DLTensorError::NullDataPointer);
        }

        // Convert shape
        if self.ndim > 0 && self.shape.is_null() {
            return Err(DLTensorError::InvalidShape {
                ndim: self.ndim,
            });
        }

        let shape: Vec<usize> = if self.ndim == 0 {
            vec![]
        } else {
            unsafe {
                std::slice::from_raw_parts(self.shape, self.ndim as usize)
            }
            .iter()
            .map(|&d| d as usize)
            .collect()
        };

        // Convert data type
        let data_type = Self::dl_dtype_to_data_type(&self.dtype)?;

        // Calculate buffer size
        let num_elements: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let element_size = data_type.size_in_bytes();
        let buffer_size =
            num_elements * element_size + self.byte_offset as usize;

        // Create Metal buffer from the data pointer
        // Note: This assumes the data pointer is already a Metal buffer pointer
        // We create a new buffer and copy the data
        let buffer = device.new_buffer_with_data(
            self.data as *const core::ffi::c_void,
            buffer_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create MetalArray
        Ok(unsafe { MetalArray::new(buffer, &shape, data_type) })
    }

    fn dl_dtype_to_data_type(
        dtype: &DLDataType
    ) -> Result<DataType, DLTensorError> {
        // Check for vectorized types
        if dtype.lanes != 1 {
            return Err(DLTensorError::VectorizedType);
        }

        // Convert code to enum for better readability
        let code_enum = match dtype.code {
            0 => DLDataTypeCode::kDLInt,
            1 => DLDataTypeCode::kDLUInt,
            2 => DLDataTypeCode::kDLFloat,
            4 => DLDataTypeCode::kDLBfloat,
            _ => {
                return Err(DLTensorError::UnsupportedDataType {
                    code: dtype.code,
                    bits: dtype.bits,
                    lanes: dtype.lanes,
                });
            },
        };

        match (code_enum, dtype.bits) {
            // Float types
            (DLDataTypeCode::kDLFloat, 16) => Ok(DataType::F16),
            (DLDataTypeCode::kDLFloat, 32) => Ok(DataType::F32),
            (DLDataTypeCode::kDLFloat, 64) => Ok(DataType::F64),
            // BFloat
            (DLDataTypeCode::kDLBfloat, 16) => Ok(DataType::BF16),
            // Signed integers
            (DLDataTypeCode::kDLInt, 4) => Ok(DataType::I4),
            (DLDataTypeCode::kDLInt, 8) => Ok(DataType::I8),
            (DLDataTypeCode::kDLInt, 16) => Ok(DataType::I16),
            (DLDataTypeCode::kDLInt, 32) => Ok(DataType::I32),
            (DLDataTypeCode::kDLInt, 64) => Ok(DataType::I64),
            // Unsigned integers
            (DLDataTypeCode::kDLUInt, 4) => Ok(DataType::U4),
            (DLDataTypeCode::kDLUInt, 8) => Ok(DataType::U8),
            (DLDataTypeCode::kDLUInt, 16) => Ok(DataType::U16),
            (DLDataTypeCode::kDLUInt, 32) => Ok(DataType::U32),
            (DLDataTypeCode::kDLUInt, 64) => Ok(DataType::U64),
            // Unsupported type
            _ => Err(DLTensorError::UnsupportedDataType {
                code: dtype.code,
                bits: dtype.bits,
                lanes: dtype.lanes,
            }),
        }
    }
}

/// Extension trait for converting uzu types to DLTensor
pub trait ToDLTensor {
    /// Convert to DLDataType
    fn to_dl_datatype(&self) -> DLDataType;
}

impl ToDLTensor for DataType {
    fn to_dl_datatype(&self) -> DLDataType {
        match self {
            DataType::BF16 => DLDataType {
                code: 4, // kDLBfloat
                bits: 16,
                lanes: 1,
            },
            DataType::F16 => DLDataType {
                code: 2, // kDLFloat
                bits: 16,
                lanes: 1,
            },
            DataType::F32 => DLDataType {
                code: 2, // kDLFloat
                bits: 32,
                lanes: 1,
            },
            DataType::F64 => DLDataType {
                code: 2, // kDLFloat
                bits: 64,
                lanes: 1,
            },
            DataType::I4 => DLDataType {
                code: 0, // kDLInt
                bits: 4,
                lanes: 1,
            },
            DataType::U4 => DLDataType {
                code: 1, // kDLUInt
                bits: 4,
                lanes: 1,
            },
            DataType::I8 => DLDataType {
                code: 0, // kDLInt
                bits: 8,
                lanes: 1,
            },
            DataType::U8 => DLDataType {
                code: 1, // kDLUInt
                bits: 8,
                lanes: 1,
            },
            DataType::I16 => DLDataType {
                code: 0, // kDLInt
                bits: 16,
                lanes: 1,
            },
            DataType::U16 => DLDataType {
                code: 1, // kDLUInt
                bits: 16,
                lanes: 1,
            },
            DataType::I32 => DLDataType {
                code: 0, // kDLInt
                bits: 32,
                lanes: 1,
            },
            DataType::U32 => DLDataType {
                code: 1, // kDLUInt
                bits: 32,
                lanes: 1,
            },
            DataType::I64 => DLDataType {
                code: 0, // kDLInt
                bits: 64,
                lanes: 1,
            },
            DataType::U64 => DLDataType {
                code: 1, // kDLUInt
                bits: 64,
                lanes: 1,
            },
        }
    }
}
