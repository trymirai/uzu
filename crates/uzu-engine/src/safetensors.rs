use std::{
    collections::{HashMap, hash_map::Entry},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use derive_more::Debug;
use serde::Deserialize;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Context, CpuBuffer, GlobalBufferCpuGpu},
    data_type::{DataType, DataTypeRepr},
    utils::{file::File, size_for_shape::size_for_shape, strict_serde::DeserializeStrictOwned},
};

#[derive(Debug, Error)]
pub enum ParameterLoaderError<B: Backend> {
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Unable to deserialize safetensors header JSON: {0}")]
    HeaderDeserialization(#[from] serde_json::Error),
    #[error("Key {0:?} not found")]
    KeyNotFound(String),
    #[error("Unable to deserialize metadata {key:?}: {source}")]
    MetadataDeserializationError {
        key: String,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "Invalid tensor {key:?}: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}"
    )]
    InvalidTensor {
        key: String,
        shape: Box<[u32]>,
        data_type: DataType,
        expected_shape: Box<[u32]>,
        expected_data_type: DataType,
    },
    #[error(
        "Invalid representation for tensor {key:?}: {shape:?} @ {data_type:?} requires {expected_size} bytes, global_offset={global_offset}, begin={begin}, end={end}"
    )]
    InvalidTensorRepresentation {
        key: String,
        shape: Box<[u32]>,
        data_type: DataType,
        expected_size: usize,
        global_offset: u64,
        begin: u64,
        end: u64,
    },
    #[error("Unread tensors: {0:?}")]
    UnreadTensors(Vec<String>),
    #[error("Duplicate tensor {0:?} across shards")]
    DuplicateTensor(String),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

#[derive(Deserialize)]
struct Header {
    #[serde(default, rename = "__metadata__")]
    metadata: HashMap<String, String>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: DataType,
    shape: Box<[u32]>,
    data_offsets: (u64, u64),
    #[serde(skip)]
    read: AtomicBool,
}

pub struct ParameterLoader<'a, B: Backend> {
    shards: Vec<Shard>,
    tensor_locations: HashMap<String, usize>,
    context: &'a B::Context,
}

struct Shard {
    file: File,
    header: Header,
    data_offset: u64,
}

#[derive(Deserialize)]
struct ShardIndex {
    #[allow(dead_code)]
    #[serde(default)]
    metadata: serde_json::Value,
    weight_map: HashMap<String, String>,
}

impl Shard {
    async fn open<B: Backend>(path: impl AsRef<Path>) -> Result<Self, ParameterLoaderError<B>> {
        let file = File::open(path).await?;
        let mut header_size = [0; 8];
        file.read_exact_at(&mut header_size, 0).await?;
        let header_size = u64::from_le_bytes(header_size);
        let mut header = vec![0; header_size as usize];
        file.read_exact_at(&mut header, 8).await?;
        let header: Header = serde_json::from_slice(&header)?;
        Ok(Self {
            file,
            header,
            data_offset: 8 + header_size,
        })
    }
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub async fn new(
        path: impl AsRef<Path>,
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let path = path.as_ref();
        let directory = path.parent().unwrap_or(Path::new("."));
        let index_path = directory.join("model.safetensors.index.json");
        let shard_paths: Vec<PathBuf> = if index_path.exists() {
            let index: ShardIndex = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let mut paths = Vec::new();
            let mut indices = HashMap::new();
            for file_name in index.weight_map.values() {
                let entry = indices.entry(file_name.clone());
                if let Entry::Vacant(vacant) = entry {
                    vacant.insert(paths.len());
                    paths.push(directory.join(file_name));
                }
            }
            paths
        } else {
            vec![path.to_path_buf()]
        };
        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut tensor_locations = HashMap::new();
        for (shard_index, shard_path) in shard_paths.iter().enumerate() {
            let shard = Shard::open(shard_path).await?;
            for key in shard.header.tensors.keys() {
                if tensor_locations.insert(key.clone(), shard_index).is_some() {
                    return Err(ParameterLoaderError::DuplicateTensor(key.clone()));
                }
            }
            shards.push(shard);
        }
        Ok(Self {
            shards,
            tensor_locations,
            context,
        })
    }

    pub fn tree(&self) -> ParameterTree<'_, B> {
        ParameterTree {
            loader: self,
            prefix: String::new(),
        }
    }

    pub fn assert_all_tensors_read(&self) -> Result<(), ParameterLoaderError<B>> {
        let mut unread = self
            .shards
            .iter()
            .flat_map(|shard| shard.header.tensors.iter())
            .filter(|(_, tensor)| !tensor.read.load(Ordering::Relaxed))
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if unread.is_empty() {
            Ok(())
        } else {
            unread.sort();
            Err(ParameterLoaderError::UnreadTensors(unread))
        }
    }

    pub fn ignore_prefix(&self, prefix: &str) {
        for shard in &self.shards {
            for (key, tensor) in &shard.header.tensors {
                if key.starts_with(prefix) {
                    tensor.read.store(true, Ordering::Relaxed);
                }
            }
        }
    }
}

pub struct ParameterTree<'a, B: Backend> {
    loader: &'a ParameterLoader<'a, B>,
    prefix: String,
}

impl<B: Backend> ParameterTree<'_, B> {
    pub fn subtree(
        &self,
        name: &str,
    ) -> Self {
        Self {
            loader: self.loader,
            prefix: format!("{}{name}.", self.prefix),
        }
    }

    pub fn metadata<T: DeserializeStrictOwned>(
        &self,
        name: &str,
    ) -> Result<T, ParameterLoaderError<B>> {
        let key = format!("{}{name}", self.prefix);
        let mut matches = self.loader.shards.iter().filter_map(|shard| shard.header.metadata.get(&key));
        let json = matches.next().ok_or_else(|| ParameterLoaderError::KeyNotFound(key.clone()))?;
        debug_assert!(matches.next().is_none(), "metadata key {key:?} appears in multiple shards",);
        serde_json::from_str(json).map_err(|source| ParameterLoaderError::MetadataDeserializationError {
            key,
            source,
        })
    }

    fn shard_and_tensor(
        &self,
        key: &str,
    ) -> Result<(&Shard, &TensorInfo), ParameterLoaderError<B>> {
        let shard_index =
            self.loader.tensor_locations.get(key).ok_or_else(|| ParameterLoaderError::KeyNotFound(key.to_string()))?;
        let shard = &self.loader.shards[*shard_index];
        let tensor = shard.header.tensors.get(key).expect("tensor location references a missing shard entry");
        Ok((shard, tensor))
    }

    fn tensor(
        &self,
        name: &str,
        shape: &[u32],
        dtype: DataType,
    ) -> Result<(&Shard, &TensorInfo), ParameterLoaderError<B>> {
        let key = format!("{}{name}", self.prefix);
        let (shard, tensor) = self.shard_and_tensor(&key)?;
        let shape_as_read = match &tensor.shape[..] {
            [1, rest @ ..] if rest == shape => rest,
            stored => stored,
        };
        if (shape_as_read, tensor.dtype) != (shape, dtype) {
            return Err(ParameterLoaderError::InvalidTensor {
                key,
                shape: tensor.shape.clone(),
                data_type: tensor.dtype,
                expected_shape: shape.into(),
                expected_data_type: dtype,
            });
        }
        let (start, end) = tensor.data_offsets;
        let expected_size = size_for_shape(shape, dtype);
        if end.checked_sub(start) != Some(expected_size as u64) || shard.data_offset.checked_add(end).is_none() {
            return Err(ParameterLoaderError::InvalidTensorRepresentation {
                key,
                shape: tensor.shape.clone(),
                data_type: tensor.dtype,
                expected_size,
                global_offset: shard.data_offset,
                begin: start,
                end,
            });
        }
        Ok((shard, tensor))
    }

    pub fn ignore(
        &self,
        name: &str,
        shape: &[u32],
        dtype: DataType,
    ) -> Result<(), ParameterLoaderError<B>> {
        let (_, tensor) = self.tensor(name, shape, dtype)?;
        tensor.read.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn read_buffer(
        &self,
        name: &str,
        shape: &[u32],
        dtype: DataType,
    ) -> Result<B::GlobalGpuBuffer, ParameterLoaderError<B>> {
        let (shard, tensor) = self.tensor(name, shape, dtype)?;
        let (start, end) = tensor.data_offsets;
        let mut buffer = self
            .loader
            .context
            .create_buffer_cpu_gpu((end - start) as usize)
            .map_err(ParameterLoaderError::BackendError)?;
        shard.file.read_exact_at(buffer.as_slice_mut::<u8>(), shard.data_offset + start).await?;
        tensor.read.store(true, Ordering::Relaxed);
        buffer.handoff().map_err(ParameterLoaderError::BackendError)
    }

    pub async fn read_slice<T: DataTypeRepr>(
        &self,
        name: &str,
        shape: &[u32],
    ) -> Result<Box<[T]>, ParameterLoaderError<B>> {
        let (shard, tensor) = self.tensor(name, shape, T::data_type())?;
        let (start, end) = tensor.data_offsets;
        let size = (end - start) as usize;
        let mut data = vec![T::zeroed(); size / size_of::<T>()];
        shard.file.read_exact_at(bytemuck::cast_slice_mut(&mut data), shard.data_offset + start).await?;
        tensor.read.store(true, Ordering::Relaxed);
        Ok(data.into_boxed_slice())
    }
}
