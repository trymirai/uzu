// This code is based on the safetensors implementation: https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html

use std::{collections::HashMap, fs::File, path::Path, str::Utf8Error};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{data_type::DataType, utils::fs::file_read_exact_at};

#[derive(Debug, Error)]
pub enum HeaderLoadingError {
    #[error("Unable to read safetensors header bytes: {0}")]
    UnableToReadHeader(#[source] std::io::Error),
    #[error("Unable to read safetensors header JSON: {0}")]
    UnableToReadHeaderJson(#[source] std::io::Error),
    #[error("The header is an invalid UTF-8 string and cannot be read: {0}")]
    InvalidHeader(#[from] Utf8Error),
    #[error("The header does contain a valid string, but it is not valid JSON: {0}")]
    InvalidHeaderDeserialization(#[from] serde_json::Error),
    #[error("The header is too large.")]
    HeaderTooLarge,
    #[error("The header length is invalid.")]
    InvalidHeaderLength,
    #[error("Unsupported safetensors dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("Invalid data offsets for tensor {key}: begin={begin}, end={end}")]
    InvalidTensorOffsets {
        key: Box<str>,
        begin: usize,
        end: usize,
    },
    #[error("Tensor offset overflow for tensor {key}: global_offset={global_offset}, local_begin={local_begin}")]
    TensorOffsetOverflow {
        key: Box<str>,
        global_offset: usize,
        local_begin: usize,
    },
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

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
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

impl Dtype {
    pub fn data_type(self) -> Result<DataType, HeaderLoadingError> {
        Ok(match self {
            Self::F16 => DataType::F16,
            Self::BF16 => DataType::BF16,
            Self::F32 => DataType::F32,
            Self::I8 => DataType::I8,
            Self::U8 => DataType::U8,
            Self::I32 => DataType::I32,
            Self::U32 => DataType::U32,
            Self::I64 => DataType::I64,
            Self::U64 => DataType::U64,
            dtype => return Err(HeaderLoadingError::UnsupportedDtype(dtype)),
        })
    }
}

const MAX_HEADER_SIZE: usize = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderSummary {
    pub tensor_count: usize,
    pub logical_payload_bytes: u64,
}

impl HeaderSummary {
    pub fn from_metadata(metadata: &HashMetadata) -> Result<Self, HeaderLoadingError> {
        let mut logical_payload_bytes = 0u64;
        for (key, tensor) in &metadata.tensors {
            let (begin, end) = tensor.data_offsets;
            let size = end.checked_sub(begin).ok_or_else(|| HeaderLoadingError::InvalidTensorOffsets {
                key: key.clone().into_boxed_str(),
                begin,
                end,
            })?;
            logical_payload_bytes =
                logical_payload_bytes.checked_add(size as u64).ok_or(HeaderLoadingError::InvalidHeaderLength)?;
        }
        Ok(Self {
            tensor_count: metadata.tensors.len(),
            logical_payload_bytes,
        })
    }
}

pub fn summarize_header(path: &Path) -> Result<HeaderSummary, HeaderLoadingError> {
    let file = File::open(path).map_err(HeaderLoadingError::UnableToReadHeader)?;
    let (_, metadata) = read_metadata(&file)?;
    HeaderSummary::from_metadata(&metadata)
}

pub fn read_metadata(file: &File) -> Result<(usize, HashMetadata), HeaderLoadingError> {
    let mut header_buffer = [0u8; size_of::<u64>()];
    file_read_exact_at(file, &mut header_buffer, 0).map_err(HeaderLoadingError::UnableToReadHeader)?;
    let metadata_size: usize =
        u64::from_le_bytes(header_buffer).try_into().map_err(|_| HeaderLoadingError::HeaderTooLarge)?;
    if metadata_size > MAX_HEADER_SIZE {
        return Err(HeaderLoadingError::InvalidHeaderLength);
    }

    let stop = metadata_size.checked_add(8).ok_or(HeaderLoadingError::InvalidHeaderLength)?;
    let mut json_buffer: Box<[u8]> = core::iter::repeat_n(0, stop - size_of::<u64>()).collect();
    file_read_exact_at(file, &mut json_buffer, 8).map_err(HeaderLoadingError::UnableToReadHeaderJson)?;
    let string = core::str::from_utf8(&json_buffer)?;
    let metadata: HashMetadata = serde_json::from_str(string)?;
    Ok((stop, metadata))
}

#[cfg(test)]
#[path = "../../unit/parameters/safetensors_metadata.rs"]
mod tests;
