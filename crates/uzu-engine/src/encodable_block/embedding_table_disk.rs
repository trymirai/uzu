use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::{
    file::{open_random_access, read_exact_at},
    size_for_shape::size_for_shape,
};
use crate::{
    backends::Backend,
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};

/// Like [`EmbeddingTable`](super::embedding_table::EmbeddingTable), but stored on disk.
///
/// Used in ngram embeddings (e.g., Qwen3.8-Flash-Next's has a ~100GB table)
#[cfg(not(target_arch = "wasm32"))]
pub struct EmbeddingTableDisk {
    file: Arc<std::fs::File>,
    range: std::range::Range<u64>,
    num_rows: u32,
    row_size: usize,
}

// Uninhabited type on wasm
#[cfg(target_arch = "wasm32")]
pub enum EmbeddingTableDisk {}

#[derive(Default)]
pub struct SharedFiles {
    #[cfg(not(target_arch = "wasm32"))]
    files: HashMap<PathBuf, Arc<std::fs::File>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SharedFiles {
    fn open(
        &mut self,
        path: &Path,
    ) -> io::Result<Arc<std::fs::File>> {
        if let Some(file) = self.files.get(path) {
            return Ok(file.clone());
        }
        let file = Arc::new(open_random_access(path)?);
        self.files.insert(path.to_path_buf(), file.clone());
        Ok(file)
    }
}

#[derive(Debug, Error)]
pub enum EmbeddingTableDiskError<B: Backend> {
    #[error("On-disk embedding tables are not supported on this platform")]
    UnsupportedPlatform,
    #[error("ParameterLoader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Failed to open table: {0}")]
    Open(#[source] io::Error),
}

#[cfg(target_arch = "wasm32")]
impl EmbeddingTableDisk {
    pub async fn load<B: Backend>(
        _: u32,
        _: u32,
        _: DataType,
        _: &ParameterTree<'_, B>,
        _: &mut SharedFiles,
    ) -> Result<Self, EmbeddingTableDiskError<B>> {
        Err(EmbeddingTableDiskError::UnsupportedPlatform)
    }
}

#[cfg(target_arch = "wasm32")]
pub fn lookup(lookups: &[(&EmbeddingTableDisk, impl AsRef<[u32]>)]) -> io::Result<Vec<Vec<u8>>> {
    match lookups.first() {
        None => Ok(Vec::new()),
        Some((table, _)) => match **table {},
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl EmbeddingTableDisk {
    pub async fn load<B: Backend>(
        num_rows: u32,
        row_dim: u32,
        data_type: DataType,
        subtree: &ParameterTree<'_, B>,
        files: &mut SharedFiles,
    ) -> Result<Self, EmbeddingTableDiskError<B>> {
        let region = subtree.claim("weights", &[num_rows, row_dim], data_type)?;
        let file = files.open(&region.path).map_err(EmbeddingTableDiskError::Open)?;

        Ok(Self {
            file,
            range: region.range,
            num_rows,
            row_size: size_for_shape(&[1, row_dim], data_type),
        })
    }

    fn offset(
        &self,
        id: u32,
    ) -> u64 {
        assert!(id < self.num_rows, "n-gram id {id} out of range for a table with {} rows", self.num_rows);
        self.range.start + id as u64 * self.row_size as u64
    }
}

/// Deduplicated lookups.
#[cfg(not(target_arch = "wasm32"))]
pub fn lookup(query: &[(&EmbeddingTableDisk, impl AsRef<[u32]>)]) -> io::Result<Vec<Vec<u8>>> {
    // Read each distinct id once per table
    let mut unique = query
        .iter()
        .map(|(table, ids)| {
            let mut unique_ids = Vec::new();
            let mut unique_index = HashMap::new();
            let indices = ids
                .as_ref()
                .iter()
                .map(|&id| {
                    *unique_index.entry(id).or_insert_with(|| {
                        unique_ids.push(id);
                        unique_ids.len() - 1
                    })
                })
                .collect::<Vec<usize>>();
            let rows = vec![0u8; unique_ids.len() * table.row_size];
            (unique_ids, indices, rows)
        })
        .collect::<Vec<_>>();

    let reads = query
        .iter()
        .zip(&mut unique)
        .flat_map(|((table, _), (unique_ids, _, unique_rows))| {
            unique_ids
                .iter()
                .zip(unique_rows.chunks_exact_mut(table.row_size))
                .map(move |(&id, row)| (&*table.file, table.offset(id), row))
        })
        .collect();

    read_exact_at(reads)?;

    let outputs = query
        .iter()
        .zip(&unique)
        .map(|((table, _), (_, indices, unique_rows))| {
            let row_size = table.row_size;
            let mut rows = Vec::with_capacity(indices.len() * row_size);
            for &index in indices {
                rows.extend_from_slice(&unique_rows[index * row_size..(index + 1) * row_size]);
            }
            rows
        })
        .collect();

    Ok(outputs)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "../../unit/encodable_block/embedding_table_disk_test.rs"]
mod tests;
