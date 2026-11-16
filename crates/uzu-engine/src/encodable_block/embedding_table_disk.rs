use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::os::unix::fs::FileExt;

use thiserror::Error;

use crate::{
    backends::Backend,
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{safetensors::TensorRegion, utils::size_for_shape::size_for_shape};

/// Like [`EmbeddingTable`](super::embedding_table::EmbeddingTable), but stored on disk.
///
/// Used in ngram embeddings (e.g., Qwen3.8-Flash-Next's has a ~100GB table)
#[cfg(not(target_arch = "wasm32"))]
pub struct EmbeddingTableDisk {
    tensor: TensorRegion,
    num_rows: u32,
    row_size: usize,
}

// Uninhabited type on wasm
#[cfg(target_arch = "wasm32")]
pub enum EmbeddingTableDisk {}

#[derive(Debug, Error)]
pub enum EmbeddingTableDiskError<B: Backend> {
    #[error("On-disk embedding tables are not supported on this platform")]
    UnsupportedPlatform,
    #[error("ParameterLoader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

#[cfg(target_arch = "wasm32")]
impl EmbeddingTableDisk {
    pub async fn load<B: Backend>(
        _: u32,
        _: u32,
        _: DataType,
        _: &ParameterTree<'_, B>,
    ) -> Result<Self, EmbeddingTableDiskError<B>> {
        Err(EmbeddingTableDiskError::UnsupportedPlatform)
    }

    pub fn lookup(
        &self,
        _: &[u32],
        _: &mut Vec<u8>,
    ) -> io::Result<()> {
        match *self {}
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl EmbeddingTableDisk {
    pub async fn load<B: Backend>(
        num_rows: u32,
        row_dim: u32,
        data_type: DataType,
        subtree: &ParameterTree<'_, B>,
    ) -> Result<Self, EmbeddingTableDiskError<B>> {
        let tensor = subtree.claim("weights", &[num_rows, row_dim], data_type).await?;

        Ok(Self {
            tensor,
            num_rows,
            row_size: size_for_shape(&[1, row_dim], data_type),
        })
    }

    pub fn lookup(
        &self,
        ngram_ids: &[u32],
        rows: &mut Vec<u8>,
    ) -> io::Result<()> {
        rows.clear();
        rows.resize(ngram_ids.len() * self.row_size, 0);
        for (&id, row) in ngram_ids.iter().zip(rows.chunks_exact_mut(self.row_size)) {
            assert!(id < self.num_rows, "n-gram id {id} out of range for a table with {} rows", self.num_rows);
            let offset = self.tensor.range.start + id as u64 * self.row_size as u64;
            self.tensor.file.read_exact_at(row, offset)?;
        }
        Ok(())
    }
}
