use std::{
    collections::HashMap,
    path::Path,
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
    file: File,
    context: &'a B::Context,
    header: Header,
    data_offset: u64,
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub async fn new(
        path: impl AsRef<Path>,
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let file = File::open(path).await?;
        let mut header_size = [0; 8];
        file.read_exact_at(&mut header_size, 0).await?;
        let header_size = u64::from_le_bytes(header_size);
        let mut header = vec![0; header_size as usize];
        file.read_exact_at(&mut header, 8).await?;
        let header: Header = serde_json::from_slice(&header)?;
        Ok(Self {
            file,
            context,
            header,
            data_offset: 8 + header_size,
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
            .header
            .tensors
            .iter()
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
        let json =
            self.loader.header.metadata.get(&key).ok_or_else(|| ParameterLoaderError::KeyNotFound(key.clone()))?;
        serde_json::from_str(json).map_err(|source| ParameterLoaderError::MetadataDeserializationError {
            key,
            source,
        })
    }

    fn tensor(
        &self,
        name: &str,
        shape: &[u32],
        dtype: DataType,
    ) -> Result<&TensorInfo, ParameterLoaderError<B>> {
        let key = format!("{}{name}", self.prefix);
        let tensor =
            self.loader.header.tensors.get(&key).ok_or_else(|| ParameterLoaderError::KeyNotFound(key.clone()))?;
        if (tensor.shape.as_ref(), tensor.dtype) != (shape, dtype) {
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
        if end.checked_sub(start) != Some(expected_size as u64) || self.loader.data_offset.checked_add(end).is_none() {
            return Err(ParameterLoaderError::InvalidTensorRepresentation {
                key,
                shape: tensor.shape.clone(),
                data_type: tensor.dtype,
                expected_size,
                global_offset: self.loader.data_offset,
                begin: start,
                end,
            });
        }
        Ok(tensor)
    }

    pub async fn read_buffer(
        &self,
        name: &str,
        shape: &[u32],
        dtype: DataType,
    ) -> Result<B::GlobalGpuBuffer, ParameterLoaderError<B>> {
        let tensor = self.tensor(name, shape, dtype)?;
        let (start, end) = tensor.data_offsets;
        let mut buffer = self
            .loader
            .context
            .create_buffer_cpu_gpu((end - start) as usize)
            .map_err(ParameterLoaderError::BackendError)?;
        self.loader.file.read_exact_at(buffer.as_slice_mut::<u8>(), self.loader.data_offset + start).await?;
        tensor.read.store(true, Ordering::Relaxed);
        Ok(buffer.handoff())
    }

    pub async fn read_slice<T: DataTypeRepr>(
        &self,
        name: &str,
        shape: &[u32],
    ) -> Result<Box<[T]>, ParameterLoaderError<B>> {
        let tensor = self.tensor(name, shape, T::data_type())?;
        let (start, end) = tensor.data_offsets;
        let size = (end - start) as usize;
        let mut data = vec![T::zeroed(); size / size_of::<T>()];
        self.loader.file.read_exact_at(bytemuck::cast_slice_mut(&mut data), self.loader.data_offset + start).await?;
        tensor.read.store(true, Ordering::Relaxed);
        Ok(data.into_boxed_slice())
    }
}
