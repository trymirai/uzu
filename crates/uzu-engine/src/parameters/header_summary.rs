use std::{fs::File, path::Path};

use super::safetensors_metadata::{HashMetadata, HeaderLoadingError, read_metadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderSummary {
    pub tensor_count: usize,
    pub logical_payload_bytes: u64,
}

impl HeaderSummary {
    pub fn read(path: &Path) -> Result<Self, HeaderLoadingError> {
        let file = File::open(path).map_err(HeaderLoadingError::UnableToReadHeader)?;
        let (_, metadata) = read_metadata(&file)?;
        Self::from_metadata(&metadata)
    }

    fn from_metadata(metadata: &HashMetadata) -> Result<Self, HeaderLoadingError> {
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
