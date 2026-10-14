// This code is based on the safetensors implementation: https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "wasi")]
use std::os::wasi::fs::FileExt;
use std::{collections::HashMap, fs::File};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data_type::DataType;

#[derive(Debug, Error)]
pub enum HeaderLoadingError {
    #[error("Unable to read safetensors header bytes: {0}")]
    UnableToReadHeader(#[source] std::io::Error),
    #[error("Unable to read safetensors header JSON: {0}")]
    UnableToReadHeaderJson(#[source] std::io::Error),
    #[error("The header is not valid UTF-8 JSON: {0}")]
    InvalidHeaderDeserialization(#[from] serde_json::Error),
    #[error("The header is too large.")]
    HeaderTooLarge,
    #[error("Unsupported safetensors dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("Invalid data offsets for tensor {key}: begin={begin}, end={end}")]
    InvalidTensorOffsets {
        key: Box<str>,
        begin: u64,
        end: u64,
    },
    #[error("Tensor offset overflow for tensor {key}: global_offset={global_offset}, local_begin={local_begin}")]
    TensorOffsetOverflow {
        key: Box<str>,
        global_offset: u64,
        local_begin: u64,
    },
    #[error("Tensor {key} extends beyond the weights data: end={end}, available={available}")]
    TensorOutOfBounds {
        key: Box<str>,
        end: u64,
        available: u64,
    },
    #[error("Tensor {key} is too large for this address space: {size} bytes")]
    TensorTooLarge {
        key: Box<str>,
        size: u64,
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
    pub shape: Vec<u32>,
    pub data_offsets: (u64, u64),
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum Dtype {
    /// Boolean type
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

pub fn read_metadata(file: &File) -> Result<(u64, HashMetadata), HeaderLoadingError> {
    let mut header_buffer = [0u8; size_of::<u64>()];
    read_exact_at(file, &mut header_buffer, 0).map_err(HeaderLoadingError::UnableToReadHeader)?;
    let metadata_size: usize =
        u64::from_le_bytes(header_buffer).try_into().map_err(|_| HeaderLoadingError::HeaderTooLarge)?;
    if metadata_size > MAX_HEADER_SIZE {
        return Err(HeaderLoadingError::HeaderTooLarge);
    }

    let mut json_buffer = vec![0u8; metadata_size];
    read_exact_at(file, &mut json_buffer, size_of::<u64>() as u64)
        .map_err(HeaderLoadingError::UnableToReadHeaderJson)?;
    let metadata: HashMetadata = serde_json::from_slice(&json_buffer)?;
    Ok(((size_of::<u64>() + metadata_size) as u64, metadata))
}

/// Reads a safetensors header directly from a downloaded or embedded model.
/// Tensor data remains borrowed; loading the header does not copy the weights.
pub fn read_metadata_from_bytes(bytes: &[u8]) -> Result<(u64, HashMetadata), HeaderLoadingError> {
    let truncated = || std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated safetensors file");
    let header: [u8; 8] =
        bytes.get(..8).ok_or_else(|| HeaderLoadingError::UnableToReadHeader(truncated()))?.try_into().unwrap();
    let size = usize::try_from(u64::from_le_bytes(header)).map_err(|_| HeaderLoadingError::HeaderTooLarge)?;
    if size > MAX_HEADER_SIZE {
        return Err(HeaderLoadingError::HeaderTooLarge);
    }
    let offset = 8 + size;
    let json = bytes.get(8..offset).ok_or_else(|| HeaderLoadingError::UnableToReadHeaderJson(truncated()))?;
    Ok((offset as u64, serde_json::from_slice(json)?))
}

pub fn read_metadata_from_source(
    source: &dyn super::ParameterReadSource
) -> Result<(u64, HashMetadata), HeaderLoadingError> {
    let mut header = [0u8; 8];
    source.read_exact_at(&mut header, 0).map_err(HeaderLoadingError::UnableToReadHeader)?;
    let size = usize::try_from(u64::from_le_bytes(header)).map_err(|_| HeaderLoadingError::HeaderTooLarge)?;
    if size > MAX_HEADER_SIZE {
        return Err(HeaderLoadingError::HeaderTooLarge);
    }
    if 8 + size as u64 > source.len() {
        return Err(HeaderLoadingError::UnableToReadHeaderJson(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Header extends beyond the weights source",
        )));
    }
    let mut json = vec![0u8; size];
    source.read_exact_at(&mut json, 8).map_err(HeaderLoadingError::UnableToReadHeaderJson)?;
    Ok((8 + size as u64, serde_json::from_slice(&json)?))
}

pub(super) fn read_exact_at(
    file: &File,
    destination: &mut [u8],
    offset: u64,
) -> std::io::Result<()> {
    #[cfg(any(unix, target_os = "wasi"))]
    return file.read_exact_at(destination, offset);

    #[cfg(not(any(unix, target_os = "wasi")))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(destination)
    }
}

#[cfg(test)]
#[path = "../../unit/parameters/safetensors_metadata.rs"]
mod tests;
