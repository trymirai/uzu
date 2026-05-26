// This code is based on the safetensors implementation: https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    io::{self, Write},
    str::Utf8Error,
};

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
            Self::I16 => DataType::I16,
            Self::U16 => DataType::U16,
            Self::I32 => DataType::I32,
            Self::U32 => DataType::U32,
            Self::F64 => DataType::F64,
            Self::I64 => DataType::I64,
            Self::U64 => DataType::U64,
            dtype => return Err(HeaderLoadingError::UnsupportedDtype(dtype)),
        })
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
            DataType::I16 => Ok(Dtype::I16),
            DataType::U16 => Ok(Dtype::U16),
            DataType::I32 => Ok(Dtype::I32),
            DataType::U32 => Ok(Dtype::U32),
            DataType::F64 => Ok(Dtype::F64),
            DataType::I64 => Ok(Dtype::I64),
            DataType::U64 => Ok(Dtype::U64),
            DataType::I4 | DataType::U4 => Err(dtype),
        }
    }
}

const MAX_HEADER_SIZE: usize = 100_000_000;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafeTensorData<'data> {
    pub name: String,
    pub shape: Box<[usize]>,
    pub data_type: DataType,
    pub data: Cow<'data, [u8]>,
}

pub fn write_safetensors<W: Write>(
    writer: &mut W,
    tensors: &[SafeTensorData<'_>],
) -> Result<(), io::Error> {
    write_safetensors_with_metadata(writer, tensors, None)
}

pub fn write_safetensors_with_metadata<W: Write>(
    writer: &mut W,
    tensors: &[SafeTensorData<'_>],
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<(), io::Error> {
    let mut sorted_tensors: Vec<&SafeTensorData<'_>> = tensors.iter().collect();
    sorted_tensors.sort_by(|left, right| {
        right.data_type.size_in_bytes().cmp(&left.data_type.size_in_bytes()).then_with(|| left.name.cmp(&right.name))
    });
    let Some(first_tensor) = sorted_tensors.first() else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "safetensors requires at least one tensor"));
    };

    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        header.insert("__metadata__".to_string(), serde_json::to_value(metadata)?);
    }

    let mut offset: usize = 0;
    let mut names = BTreeSet::new();
    for tensor in sorted_tensors.iter() {
        if tensor.name == "__metadata__" {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "__metadata__ is reserved"));
        }
        if !names.insert(tensor.name.as_str()) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "duplicate safetensors tensor name"));
        }
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
        let end = offset
            .checked_add(tensor.data.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "safetensors tensor offsets overflow usize"))?;
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
        writer.write_all(tensor.data.as_ref())?;
    }
    Ok(())
}
