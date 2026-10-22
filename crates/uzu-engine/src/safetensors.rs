#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_family = "wasm")]
use std::os::wasi::fs::FileExt;
use std::{cell::Cell, collections::HashMap, fs::File};

use derive_more::Debug;
use serde::Deserialize;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferMut, Context},
    data_type::{DataType, DataTypeRepr},
    utils::{size_for_shape::size_for_shape, strict_serde::DeserializeStrictOwned},
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
    read: Cell<bool>,
}

pub struct ParameterLoader<'a, B: Backend> {
    file: &'a File,
    context: &'a B::Context,
    header: Header,
    data_offset: u64,
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub fn new(
        file: &'a File,
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let mut header_size = [0; 8];
        file.read_exact_at(&mut header_size, 0)?;
        let header_size = u64::from_le_bytes(header_size);
        let mut header = vec![0; header_size as usize];
        file.read_exact_at(&mut header, 8)?;
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
            unit_mixture: false,
        }
    }

    pub fn assert_all_tensors_read(&self) -> Result<(), ParameterLoaderError<B>> {
        let mut unread = self
            .header
            .tensors
            .iter()
            .filter(|(_, tensor)| !tensor.read.get())
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
    unit_mixture: bool,
}

impl<B: Backend> ParameterTree<'_, B> {
    pub fn subtree(
        &self,
        name: &str,
    ) -> Self {
        Self {
            loader: self.loader,
            prefix: format!("{}{name}.", self.prefix),
            unit_mixture: self.unit_mixture,
        }
    }

    #[allow(dead_code)]
    pub fn unit_mixture(&self) -> Self {
        Self {
            loader: self.loader,
            prefix: self.prefix.clone(),
            unit_mixture: true,
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
        let matches_shape = tensor.shape.as_ref() == shape
            || (self.unit_mixture && tensor.shape.first() == Some(&1) && &tensor.shape[1..] == shape);
        if !matches_shape || tensor.dtype != dtype {
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

    pub fn read_buffer(
        &self,
        name: &str,
        shape: &[u32],
        dtype: DataType,
    ) -> Result<B::GlobalBuffer, ParameterLoaderError<B>> {
        let tensor = self.tensor(name, shape, dtype)?;
        let (start, end) = tensor.data_offsets;
        let mut buffer =
            self.loader.context.create_buffer((end - start) as usize).map_err(ParameterLoaderError::BackendError)?;
        self.loader.file.read_exact_at(buffer.as_slice_mut::<u8>(), self.loader.data_offset + start)?;
        tensor.read.set(true);
        Ok(buffer)
    }

    pub fn read_slice<T: DataTypeRepr>(
        &self,
        name: &str,
        shape: &[u32],
    ) -> Result<Box<[T]>, ParameterLoaderError<B>> {
        let tensor = self.tensor(name, shape, T::data_type())?;
        let (start, end) = tensor.data_offsets;
        let size = (end - start) as usize;
        let mut data = vec![T::zeroed(); size / size_of::<T>()];
        self.loader.file.read_exact_at(bytemuck::cast_slice_mut(&mut data), self.loader.data_offset + start)?;
        tensor.read.set(true);
        Ok(data.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use futures::executor::block_on;
    use uzu_engine_macros::uzu_test;

    use super::*;
    use crate::backends::{common::Context, cpu::Cpu};

    #[uzu_test]
    fn test_unit_mixture_tensor_reading() {
        let temp_path = std::env::temp_dir().join(format!(
            "safetensors_test_{}_{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        let header_json = serde_json::json!({
            "shared_weight": {
                "dtype": "F32",
                "shape": [1, 4, 8],
                "data_offsets": [0, 128]
            }
        })
        .to_string();

        let header_bytes = header_json.as_bytes();
        let header_len = (header_bytes.len() as u64).to_le_bytes();
        let raw_data = vec![0u8; 128];

        {
            let mut file = std::fs::File::create(&temp_path).expect("failed to create temp file");
            file.write_all(&header_len).unwrap();
            file.write_all(header_bytes).unwrap();
            file.write_all(&raw_data).unwrap();
            file.flush().unwrap();
        }

        let std_file = std::fs::File::open(&temp_path).expect("failed to open temp file");
        let context = block_on(<Cpu as Backend>::Context::new()).expect("failed to create Cpu context");
        let loader = ParameterLoader::<Cpu>::new(&std_file, context.as_ref()).expect("failed to create loader");

        // Standard reading without unit_mixture: reading [4, 8] must fail because the file has [1, 4, 8]
        let default_tree = loader.tree();
        assert!(default_tree.read_slice::<f32>("shared_weight", &[4, 8]).is_err());

        // Opt-in unit_mixture: reading [4, 8] must succeed by accepting the leading dimension of 1
        let mixture_tree = default_tree.unit_mixture();
        let slice = mixture_tree.read_slice::<f32>("shared_weight", &[4, 8]).expect("should succeed with unit_mixture");
        assert_eq!(slice.len(), 32);

        let _ = std::fs::remove_file(temp_path);
    }
}
