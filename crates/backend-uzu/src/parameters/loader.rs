use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use thiserror::Error;

use super::safetensors_metadata::{HeaderLoadingError, read_metadata as read_st_metadata};
use crate::{
    array::{ArrayElement, size_for_shape},
    backends::common::{Allocation, AllocationType, AsBufferRangeRef, Backend, Context, DenseBuffer},
    data_type::DataType,
    utils::{fs::file_read_exact_at, strict_serde::DeserializeStrictOwned},
};

pub struct ParameterMetadata {
    shape: Box<[usize]>,
    data_type: DataType,
    offset: usize,
    size: usize,
}

#[derive(Debug, Error)]
pub enum ParameterLoaderError<B: Backend> {
    #[error("Array with key \"{0}\" not found.")]
    KeyNotFound(String),
    #[error("Couldn't find any arrays with prefix \"{0}\".")]
    SubtreeNotFound(String),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Failed to read data")]
    ArrayLoadingError(#[from] std::io::Error),
    #[error("Row loading is only available for file-backed parameters")]
    RowSourceUnavailable,
    #[error("Failed to deserialize metadata")]
    MetadataDeserializationError(#[from] serde_json::Error),
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[usize]>,
        data_type: DataType,
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
    },
    #[error("Invalid tensor byte size: got {size} bytes for {shape:?} @ {data_type:?}, expected {expected_size} bytes")]
    InvalidTensorSize {
        shape: Box<[usize]>,
        data_type: DataType,
        size: usize,
        expected_size: usize,
    },
    #[error("Unvalidated tensors under {prefix:?}: {keys:?}")]
    UnvalidatedTensors {
        prefix: Option<String>,
        keys: Box<[String]>,
    },
}

pub struct ParameterRowSource {
    file: File,
    offset: u64,
    row_bytes: usize,
    row_count: usize,
}

impl ParameterRowSource {
    pub fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self {
            file: self.file.try_clone()?,
            offset: self.offset,
            row_bytes: self.row_bytes,
            row_count: self.row_count,
        })
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn read_rows(
        &self,
        row_ids: &[u64],
        destination: &mut [u8],
    ) -> std::io::Result<()> {
        let expected_size = row_ids
            .len()
            .checked_mul(self.row_bytes)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "row staging size overflow"))?;
        if destination.len() != expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "row staging destination has the wrong size",
            ));
        }

        for (row_index, &row_id) in row_ids.iter().enumerate() {
            let row_id = usize::try_from(row_id)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "row ID does not fit in usize"))?;
            if row_id >= self.row_count {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "row ID is out of range"));
            }
            let row_offset = self
                .offset
                .checked_add((row_id * self.row_bytes) as u64)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "row offset overflow"))?;
            let destination_offset = row_index * self.row_bytes;
            crate::utils::fs::file_read_exact_at(
                &self.file,
                &mut destination[destination_offset..destination_offset + self.row_bytes],
                row_offset,
            )?;
        }
        Ok(())
    }
}

enum ParameterBytes<'a> {
    File(&'a File),
    Random(u64),
}

pub struct ParameterLoader<'a, B: Backend> {
    context: &'a B::Context,
    index: HashMap<String, ParameterMetadata>,
    metadata: HashMap<String, String>,
    validated_tensors: RefCell<HashSet<String>>,
    bytes: ParameterBytes<'a>,
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub fn new(
        file: &'a File,
        context: &'a B::Context,
    ) -> Result<Self, HeaderLoadingError> {
        Self::from_header(file, context, ParameterBytes::File(file))
    }

    pub fn new_random(
        header_file: &File,
        context: &'a B::Context,
        seed: u64,
    ) -> Result<Self, HeaderLoadingError> {
        Self::from_header(header_file, context, ParameterBytes::Random(seed))
    }

    fn from_header(
        header_file: &File,
        context: &'a B::Context,
        bytes: ParameterBytes<'a>,
    ) -> Result<Self, HeaderLoadingError> {
        let (global_offset, st_metadata) = read_st_metadata(header_file)?;
        let index = st_metadata
            .tensors
            .into_iter()
            .map(|(key, value)| {
                let (local_begin, local_end) = value.data_offsets;
                let size =
                    local_end.checked_sub(local_begin).ok_or_else(|| HeaderLoadingError::InvalidTensorOffsets {
                        key: key.clone().into_boxed_str(),
                        begin: local_begin,
                        end: local_end,
                    })?;
                let offset =
                    global_offset.checked_add(local_begin).ok_or_else(|| HeaderLoadingError::TensorOffsetOverflow {
                        key: key.clone().into_boxed_str(),
                        global_offset,
                        local_begin,
                    })?;
                let weight_metadata = ParameterMetadata {
                    shape: value.shape.into(),
                    data_type: value.dtype.data_type()?,
                    offset,
                    size,
                };
                Ok((key, weight_metadata))
            })
            .collect::<Result<HashMap<_, _>, HeaderLoadingError>>()?;
        let metadata = st_metadata.metadata.unwrap_or_default();
        Ok(ParameterLoader {
            context,
            index,
            metadata,
            validated_tensors: RefCell::new(HashSet::new()),
            bytes,
        })
    }

    pub fn tree(&self) -> ParameterTree<'_, B> {
        ParameterTree {
            loader: self,
            prefix: None,
        }
    }
}

pub struct ParameterLeaf<'a, 'leaf, B: Backend, const VALIDATED: bool> {
    key: &'leaf str,
    metadata: &'leaf ParameterMetadata,
    loader: &'leaf ParameterLoader<'a, B>,
}

impl<'a, 'leaf, B: Backend> ParameterLeaf<'a, 'leaf, B, false> {
    pub fn validate(
        self,
        expected_shape: &[usize],
        expected_data_type: DataType,
    ) -> Result<ParameterLeaf<'a, 'leaf, B, true>, ParameterLoaderError<B>> {
        let shape = self.metadata.shape.as_ref();
        let data_type = self.metadata.data_type;
        if (shape, data_type) != (expected_shape, expected_data_type) {
            return Err(ParameterLoaderError::InvalidTensor {
                shape: shape.into(),
                data_type,
                expected_shape: expected_shape.into(),
                expected_data_type,
            });
        }
        let expected_size = size_for_shape(expected_shape, expected_data_type);
        if self.metadata.size != expected_size {
            return Err(ParameterLoaderError::InvalidTensorSize {
                shape: shape.into(),
                data_type,
                size: self.metadata.size,
                expected_size,
            });
        }
        self.loader.validated_tensors.borrow_mut().insert(self.key.to_string());
        Ok(ParameterLeaf {
            key: self.key,
            metadata: self.metadata,
            loader: self.loader,
        })
    }
}

impl<'a, 'leaf, B: Backend> ParameterLeaf<'a, 'leaf, B, true> {
    pub fn row_source(
        &self,
        row_bytes: usize,
        row_file: Option<&Path>,
        disable_page_cache_mode: bool,
    ) -> Result<ParameterRowSource, ParameterLoaderError<B>> {
        let ParameterBytes::File(file) = &self.loader.bytes else {
            return Err(ParameterLoaderError::RowSourceUnavailable);
        };
        if row_bytes == 0 || self.metadata.size % row_bytes != 0 {
            return Err(ParameterLoaderError::InvalidTensorSize {
                shape: self.metadata.shape.clone(),
                data_type: self.metadata.data_type,
                size: self.metadata.size,
                expected_size: row_bytes,
            });
        }
        let (file, offset) = if let Some(path) = row_file {
            let file = File::open(path)?;
            let size = file.metadata()?.len() as usize;
            if size != self.metadata.size {
                return Err(ParameterLoaderError::InvalidTensorSize {
                    shape: self.metadata.shape.clone(),
                    data_type: self.metadata.data_type,
                    size,
                    expected_size: self.metadata.size,
                });
            }
            (file, 0)
        } else {
            (file.try_clone()?, self.metadata.offset as u64)
        };
        if disable_page_cache_mode {
            disable_page_cache(&file)?;
        }
        Ok(ParameterRowSource {
            file,
            offset,
            row_bytes,
            row_count: self.metadata.size / row_bytes,
        })
    }

    pub fn read_slice<T: ArrayElement>(&self) -> Result<Box<[T]>, ParameterLoaderError<B>> {
        let element_count = self.metadata.size / std::mem::size_of::<T>();
        let mut data = vec![T::zeroed(); element_count];
        let destination = bytemuck::cast_slice_mut(&mut data);
        match &self.loader.bytes {
            ParameterBytes::File(file) => {
                file_read_exact_at(file, destination, self.metadata.offset as u64)?;
            },
            ParameterBytes::Random(seed) => fill_random(destination, self.metadata.data_type, *seed),
        }
        Ok(data.into_boxed_slice())
    }

    pub fn read_allocation(&self) -> Result<Allocation<B>, ParameterLoaderError<B>> {
        let allocation = self
            .loader
            .context
            .create_allocation(self.metadata.size, AllocationType::Global)
            .map_err(ParameterLoaderError::BackendError)?;
        let buffer_range = allocation.as_buffer_range_ref();
        let range = buffer_range.range();
        let destination = unsafe {
            std::slice::from_raw_parts_mut(
                (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start),
                range.len(),
            )
        };
        match &self.loader.bytes {
            ParameterBytes::File(file) => {
                file_read_exact_at(file, destination, self.metadata.offset as u64)?;
            },
            ParameterBytes::Random(seed) => fill_random(destination, self.metadata.data_type, *seed),
        }
        Ok(allocation)
    }
}

fn disable_page_cache(file: &File) -> std::io::Result<()> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use std::os::fd::AsRawFd;

        const F_NOCACHE: libc::c_int = 48;
        const F_RDAHEAD: libc::c_int = 45;
        const F_GLOBAL_NOCACHE: libc::c_int = 55;
        let fd = file.as_raw_fd();
        if unsafe { libc::fcntl(fd, F_GLOBAL_NOCACHE, 1) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(fd, F_NOCACHE, 1) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(fd, F_RDAHEAD, 0) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let _ = file;
    Ok(())
}

fn fill_random(
    destination: &mut [u8],
    data_type: DataType,
    seed: u64,
) {
    if destination.is_empty() {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let head_len = destination.len().min(65536);
    let (head, tail) = destination.split_at_mut(head_len);
    match data_type {
        DataType::BF16 => {
            for chunk in head.as_chunks_mut::<2>().0 {
                *chunk = bf16::from_f32(rng.random_range(-0.1f32..0.1f32)).to_le_bytes();
            }
        },
        DataType::F16 => {
            for chunk in head.as_chunks_mut::<2>().0 {
                *chunk = f16::from_f32(rng.random_range(-0.1f32..0.1f32)).to_le_bytes();
            }
        },
        DataType::F32 => {
            for chunk in head.as_chunks_mut::<4>().0 {
                *chunk = rng.random_range(-0.1f32..0.1f32).to_le_bytes();
            }
        },
        DataType::F64 => {
            for chunk in head.as_chunks_mut::<8>().0 {
                *chunk = rng.random_range(-0.1f64..0.1f64).to_le_bytes();
            }
        },
        _ => {
            for chunk in head.chunks_mut(8) {
                let bytes = rng.random::<u64>().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        },
    }
    for target in tail.chunks_mut(head_len) {
        target.copy_from_slice(&head[..target.len()]);
    }
}

pub struct ParameterTree<'loader, B: Backend> {
    loader: &'loader ParameterLoader<'loader, B>,
    prefix: Option<String>,
}

impl<'loader, B: Backend> ParameterTree<'loader, B> {
    fn join_prefix(
        &self,
        name: &str,
    ) -> String {
        self.prefix.as_ref().map_or_else(|| name.to_string(), |p| format!("{p}.{name}"))
    }

    pub fn subtree(
        &self,
        name: &str,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let new_prefix = self.join_prefix(name);
        let subtree_prefix = format!("{new_prefix}.");
        if self.loader.index.keys().any(|key| key.starts_with(&subtree_prefix)) {
            Ok(Self {
                loader: self.loader,
                prefix: Some(new_prefix),
            })
        } else {
            Err(ParameterLoaderError::SubtreeNotFound(new_prefix))
        }
    }

    pub fn leaf<'leaf>(
        &'leaf self,
        name: &str,
    ) -> Result<ParameterLeaf<'loader, 'leaf, B, false>, ParameterLoaderError<B>> {
        let key = self.join_prefix(name);
        let Some((key, metadata)) = self.loader.index.get_key_value(&key) else {
            return Err(ParameterLoaderError::KeyNotFound(key));
        };
        Ok(ParameterLeaf {
            key,
            metadata,
            loader: self.loader,
        })
    }

    pub fn metadata<T: DeserializeStrictOwned>(
        &self,
        name: &str,
    ) -> Result<T, ParameterLoaderError<B>> {
        let new_prefix = self.join_prefix(name);

        Ok(serde_json::from_str(
            self.loader.metadata.get(&new_prefix).ok_or(ParameterLoaderError::KeyNotFound(new_prefix))?,
        )?)
    }

    pub fn assert_all_tensors_validated(&self) -> Result<(), ParameterLoaderError<B>> {
        let subtree_prefix = self.prefix.as_ref().map(|prefix| format!("{prefix}."));
        let validated_tensors = self.loader.validated_tensors.borrow();
        let mut unvalidated_tensors = self
            .loader
            .index
            .keys()
            .filter(|key| subtree_prefix.as_ref().is_none_or(|prefix| key.starts_with(prefix)))
            .filter(|key| !validated_tensors.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        unvalidated_tensors.sort();
        if unvalidated_tensors.is_empty() {
            Ok(())
        } else {
            Err(ParameterLoaderError::UnvalidatedTensors {
                prefix: self.prefix.clone(),
                keys: unvalidated_tensors.into_boxed_slice(),
            })
        }
    }
}
