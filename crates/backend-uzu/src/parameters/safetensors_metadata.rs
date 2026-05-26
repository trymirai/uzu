// This code is based on the safetensors implementation: https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html

use std::{
    collections::HashMap,
    fs::File,
    io::{self, Write},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{DataType, utils::fs::file_read_exact_at};

#[derive(Debug, Error)]
pub enum HeaderLoadingError {
    #[error("The header is an invalid UTF-8 string and cannot be read.")]
    InvalidHeader,
    #[error("The header does contain a valid string, but it is not valid JSON.")]
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

impl TryFrom<DataType> for Dtype {
    type Error = DataType;

    fn try_from(dtype: DataType) -> Result<Self, Self::Error> {
        match dtype {
            DataType::F16 => Ok(Dtype::F16),
            DataType::BF16 => Ok(Dtype::BF16),
            DataType::F32 => Ok(Dtype::F32),
            DataType::I8 => Ok(Dtype::I8),
            DataType::U8 => Ok(Dtype::U8),
            DataType::I32 => Ok(Dtype::I32),
            DataType::U32 => Ok(Dtype::U32),
            DataType::I64 => Ok(Dtype::I64),
            DataType::U64 => Ok(Dtype::U64),
            DataType::F64 | DataType::I4 | DataType::U4 | DataType::I16 | DataType::U16 => Err(dtype),
        }
    }
}

const MAX_HEADER_SIZE: usize = 100_000_000;

pub fn read_metadata(file: &File) -> Result<(usize, HashMetadata), HeaderLoadingError> {
    let mut header_buffer = [0u8; size_of::<u64>()];
    file_read_exact_at(file, &mut header_buffer, 0).map_err(|_| HeaderLoadingError::HeaderTooSmall)?;
    let metadata_size: usize =
        u64::from_le_bytes(header_buffer).try_into().map_err(|_| HeaderLoadingError::HeaderTooLarge)?;
    if metadata_size > MAX_HEADER_SIZE {
        return Err(HeaderLoadingError::InvalidHeaderLength);
    }

    let stop = metadata_size.checked_add(8).ok_or(HeaderLoadingError::InvalidHeaderLength)?;
    let mut json_buffer: Box<[u8]> = core::iter::repeat(0).take(stop - size_of::<u64>()).collect();
    file_read_exact_at(file, &mut json_buffer, 8).map_err(|_| HeaderLoadingError::InvalidHeader)?;
    let string = core::str::from_utf8(&json_buffer).map_err(|_| HeaderLoadingError::InvalidHeader)?;
    let metadata: HashMetadata =
        serde_json::from_str(string).map_err(|_| HeaderLoadingError::InvalidHeaderDeserialization)?;
    Ok((stop, metadata))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafeTensorData {
    pub name: String,
    pub shape: Box<[usize]>,
    pub data_type: DataType,
    pub data: Box<[u8]>,
}

pub fn write_safetensors<W: Write>(
    writer: &mut W,
    tensors: &[SafeTensorData],
) -> Result<(), io::Error> {
    let mut sorted_tensors: Vec<&SafeTensorData> = tensors.iter().collect();
    sorted_tensors.sort_by(|left, right| {
        right.data_type.size_in_bytes().cmp(&left.data_type.size_in_bytes()).then_with(|| left.name.cmp(&right.name))
    });
    let Some(first_tensor) = sorted_tensors.first() else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "safetensors requires at least one tensor"));
    };

    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for tensor in sorted_tensors.iter() {
        let dtype = Dtype::try_from(tensor.data_type)
            .map_err(|dtype| io::Error::new(io::ErrorKind::InvalidInput, format!("unsupported dtype: {dtype:?}")))?;
        let expected_len = tensor
            .shape
            .iter()
            .try_fold(tensor.data_type.size_in_bytes(), |size, dim| size.checked_mul(*dim))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "safetensors tensor byte size overflows usize")
            })?;
        if expected_len != tensor.data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "safetensors tensor shape does not match data"));
        }
        let end = offset + tensor.data.len();
        header.insert(
            tensor.name.clone(),
            serde_json::to_value(TensorInfo {
                dtype,
                shape: tensor.shape.to_vec(),
                data_offsets: (offset, end),
            })?,
        );
        offset = end;
    }

    let data_alignment = first_tensor.data_type.size_in_bytes().max(8);
    let mut header_bytes = serde_json::to_vec(&header)?;
    let padding = (data_alignment - header_bytes.len() % data_alignment) % data_alignment;
    header_bytes.extend(std::iter::repeat_n(b' ', padding));

    let header_len = u64::try_from(header_bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "safetensors header length does not fit into u64"))?;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    for tensor in sorted_tensors {
        writer.write_all(&tensor.data)?;
    }
    Ok(())
}
