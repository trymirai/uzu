// This code is based on the safetensors implementation: https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html

use std::{collections::HashMap, fs::File, os::unix::fs::FileExt};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::DataType;

#[derive(Debug, Error)]
pub enum HeaderLoadingError {
    #[error("The header is an invalid UTF-8 string and cannot be read.")]
    InvalidHeader,
    #[error(
        "The header does contain a valid string, but it is not valid JSON."
    )]
    InvalidHeaderDeserialization,
    #[error("The header is large than 100Mo which is considered too large.")]
    HeaderTooLarge,
    #[error("The header is smaller than 8 bytes.")]
    HeaderTooSmall,
    #[error("The header length is invalid.")]
    InvalidHeaderLength,
    #[error("Couldn't deserialize the header JSON.")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(
    Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd,
)]
#[non_exhaustive]
pub enum Dtype {
    /// Boolan type
    BOOL,
    /// Unsigned byte
    U8,
    /// Signed byte
    I8,
    /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    #[allow(non_camel_case_types)]
    F8_E5M2,
    /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    #[allow(non_camel_case_types)]
    F8_E4M3,
    /// Signed integer (16-bit)
    I16,
    /// Unsigned integer (16-bit)
    U16,
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
}

impl From<Dtype> for DataType {
    fn from(val: Dtype) -> Self {
        match val {
            Dtype::F16 => DataType::F16,
            Dtype::BF16 => DataType::BF16,
            Dtype::F32 => DataType::F32,
            Dtype::I8 => DataType::I8,
            Dtype::U8 => DataType::U8,
            Dtype::I32 => DataType::I32,
            Dtype::U32 => DataType::U32,
            Dtype::I64 => DataType::I64,
            Dtype::U64 => DataType::U64,
            Dtype::BOOL => DataType::I8,
            // Add mappings for other types as needed
            _ => panic!("Unsupported dtype: {:?}", val),
        }
    }
}

impl From<DataType> for Dtype {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::F16 => Dtype::F16,
            DataType::BF16 => Dtype::BF16,
            DataType::F32 => Dtype::F32,
            DataType::I8 => Dtype::I8,
            DataType::I32 => Dtype::I32,
            DataType::I64 => Dtype::I64,
            DataType::U64 => Dtype::U64,
            _ => panic!("Unsupported dtype: {:?}", dtype),
        }
    }
}

const MAX_HEADER_SIZE: usize = 100_000_000;

pub fn read_metadata(
    file: &File
) -> Result<(usize, HashMetadata), HeaderLoadingError> {
    let mut header_buffer = [0u8; size_of::<u64>()];
    file.read_exact_at(&mut header_buffer, 0)
        .map_err(|_| HeaderLoadingError::HeaderTooSmall)?;
    let metadata_size: usize = u64::from_le_bytes(header_buffer)
        .try_into()
        .map_err(|_| HeaderLoadingError::HeaderTooLarge)?;
    if metadata_size > MAX_HEADER_SIZE {
        return Err(HeaderLoadingError::InvalidHeaderLength);
    }

    let stop = metadata_size
        .checked_add(8)
        .ok_or(HeaderLoadingError::InvalidHeaderLength)?;
    let mut json_buffer: Box<[u8]> =
        core::iter::repeat(0).take(stop - size_of::<u64>()).collect();
    file.read_exact_at(&mut json_buffer, 8)
        .map_err(|_| HeaderLoadingError::InvalidHeader)?;
    let string = core::str::from_utf8(&json_buffer)
        .map_err(|_| HeaderLoadingError::InvalidHeader)?;
    let metadata: HashMetadata = serde_json::from_str(string)
        .map_err(|_| HeaderLoadingError::InvalidHeaderDeserialization)?;
    Ok((stop, metadata))
}
