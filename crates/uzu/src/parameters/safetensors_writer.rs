use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use thiserror::Error;

use super::safetensors_metadata::{Dtype, HashMetadata, TensorInfo};
use crate::DataType;

#[derive(Debug, Error)]
pub enum SafetensorsWriteError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid tensor data for \"{name}\": expected {expected} bytes, got {actual} bytes")]
    InvalidTensorData {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Failed to serialize safetensors header: {0}")]
    HeaderJson(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy)]
pub struct SafetensorView<'a> {
    pub name: &'a str,
    pub dtype: DataType,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct SafetensorHeaderEntry {
    pub name: String,
    pub dtype: DataType,
    pub shape: Box<[usize]>,
    pub byte_len: usize,
}

pub fn write_safetensors(
    path: &Path,
    tensors: &[SafetensorView<'_>],
    metadata: Option<HashMap<String, String>>,
) -> Result<(), SafetensorsWriteError> {
    let mut offset: usize = 0;
    let mut header = HashMetadata {
        metadata,
        tensors: HashMap::new(),
    };

    for t in tensors {
        let elem_bytes = t.dtype.size_in_bytes();
        let numel: usize = t.shape.iter().product();
        let expected_bytes = numel.saturating_mul(elem_bytes);
        let actual_bytes = t.data.len();
        if expected_bytes != actual_bytes {
            return Err(SafetensorsWriteError::InvalidTensorData {
                name: t.name.to_string(),
                expected: expected_bytes,
                actual: actual_bytes,
            });
        }

        let begin = offset;
        let end = offset + actual_bytes;
        offset = end;

        header.tensors.insert(
            t.name.to_string(),
            TensorInfo {
                dtype: Dtype::from(t.dtype),
                shape: t.shape.to_vec(),
                data_offsets: (begin, end),
            },
        );
    }

    let mut header_bytes = serde_json::to_vec(&header)?;

    // Safetensors headers are typically padded to 8 bytes for alignment.
    let padding = (8 - (header_bytes.len() % 8)) % 8;
    if padding != 0 {
        header_bytes.extend(std::iter::repeat(b' ').take(padding));
    }

    let header_len: u64 = header_bytes
        .len()
        .try_into()
        .expect("header too large for u64");

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_bytes)?;
    for t in tensors {
        w.write_all(t.data)?;
    }
    w.flush()?;
    Ok(())
}

pub fn write_safetensors_streaming<F>(
    path: &Path,
    tensors: &[SafetensorHeaderEntry],
    metadata: Option<HashMap<String, String>>,
    mut write_tensor_data: F,
) -> Result<(), SafetensorsWriteError>
where
    F: FnMut(&mut dyn Write, &SafetensorHeaderEntry) -> Result<(), SafetensorsWriteError>,
{
    let mut offset: usize = 0;
    let mut header = HashMetadata {
        metadata,
        tensors: HashMap::new(),
    };

    for t in tensors {
        let elem_bytes = t.dtype.size_in_bytes();
        let numel: usize = t.shape.iter().product();
        let expected_bytes = numel.saturating_mul(elem_bytes);
        if expected_bytes != t.byte_len {
            return Err(SafetensorsWriteError::InvalidTensorData {
                name: t.name.clone(),
                expected: expected_bytes,
                actual: t.byte_len,
            });
        }

        let begin = offset;
        let end = offset + t.byte_len;
        offset = end;

        header.tensors.insert(
            t.name.clone(),
            TensorInfo {
                dtype: Dtype::from(t.dtype),
                shape: t.shape.to_vec(),
                data_offsets: (begin, end),
            },
        );
    }

    let mut header_bytes = serde_json::to_vec(&header)?;

    let padding = (8 - (header_bytes.len() % 8)) % 8;
    if padding != 0 {
        header_bytes.extend(std::iter::repeat(b' ').take(padding));
    }

    let header_len: u64 = header_bytes
        .len()
        .try_into()
        .expect("header too large for u64");

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_bytes)?;
    for t in tensors {
        write_tensor_data(&mut w, t)?;
    }
    w.flush()?;
    Ok(())
}

