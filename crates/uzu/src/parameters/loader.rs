use std::{
    collections::{HashMap, hash_map::Keys},
    fs::File,
    os::unix::fs::FileExt,
};

use thiserror::Error;

use super::safetensors_metadata::{
    HashMetadata as STMetadata, HeaderLoadingError,
    read_metadata as read_st_metadata,
};
use crate::{Array, DataType, DeviceContext};

pub struct ParameterMetadata {
    shape: Box<[usize]>,
    data_type: DataType,
    offset: usize,
    size: usize,
}

fn st_metadata_into_index(
    global_offset: usize,
    st_metadata: STMetadata,
) -> HashMap<String, ParameterMetadata> {
    st_metadata
        .tensors
        .into_iter()
        .map(|(key, value)| {
            let (local_begin, local_end) = value.data_offsets;
            let actual_local_offset = local_begin;
            let actual_size = local_end - local_begin;
            let weight_metadata = ParameterMetadata {
                shape: value.shape.into(),
                data_type: value.dtype.into(),
                offset: global_offset + actual_local_offset,
                size: actual_size,
            };
            (key, weight_metadata)
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum ParameterLoaderError {
    #[error("Array with key \"{0}\" not found.")]
    KeyNotFound(String),
    #[error("Couldn't find any arrays with prefix \"{0}\".")]
    SubtreeNotFound(String),
    #[error(
        "Size mismatch: array of shape {shape:?} and data type \
        {data_type:?} expected to be {expected_size} bytes, got {actual_size} bytes."
    )]
    SizeMismatch {
        data_type: DataType,
        shape: Box<[usize]>,
        expected_size: usize,
        actual_size: usize,
    },
    #[error("Failed to read data")]
    ArrayLoadingError(#[from] std::io::Error),
}

pub struct ParameterLoader<'context, 'file, C: DeviceContext>
where
    'file: 'context,
{
    context: &'context C,
    index: HashMap<String, ParameterMetadata>,
    file: &'file File,
}

impl<'file, 'context, C: DeviceContext> ParameterLoader<'file, 'context, C>
where
    'file: 'context,
{
    pub fn new(
        file: &'file File,
        context: &'context C,
    ) -> Result<Self, HeaderLoadingError> {
        let (global_offset, st_metadata) = read_st_metadata(file)?;
        let index = st_metadata_into_index(global_offset, st_metadata);
        Ok(ParameterLoader {
            context,
            file,
            index,
        })
    }

    pub fn keys(&self) -> Keys<'_, String, ParameterMetadata> {
        self.index.keys()
    }

    pub fn get(
        &self,
        key: &str,
    ) -> Result<C::DeviceArray, ParameterLoaderError> {
        let metadata_entry = self
            .index
            .get(key)
            .ok_or(ParameterLoaderError::KeyNotFound(key.to_string()))?;
        let (offset, size) = (metadata_entry.offset, metadata_entry.size);
        let array_key = key.replace(".", "_");
        let array_label = format!("parameter_loader_{array_key}");
        let mut array = self.context.array(
            &metadata_entry.shape,
            metadata_entry.data_type,
            array_label,
        );
        let expected_size = array.size_in_bytes();
        if expected_size != size {
            return Err(ParameterLoaderError::SizeMismatch {
                data_type: metadata_entry.data_type,
                shape: metadata_entry.shape.to_owned(),
                expected_size,
                actual_size: size,
            });
        }
        self.file.read_exact_at(array.buffer_mut(), offset as u64)?;
        Ok(array)
    }

    pub fn read_extract_at(
        &self,
        key: &str,
        buf: &mut [u8],
        shape: &mut Box<[usize]>,
        data_type: &mut DataType,
    ) -> Result<(), ParameterLoaderError> {
        let metadata_entry = self
            .index
            .get(key)
            .ok_or(ParameterLoaderError::KeyNotFound(key.to_string()))?;
        self.file.read_exact_at(buf, metadata_entry.offset as u64)?;
        *shape = metadata_entry.shape.to_owned();
        *data_type = metadata_entry.data_type;
        Ok(())
    }

    pub fn tree<'loader>(&'loader self) -> ParameterTree<'loader, C> {
        ParameterTree {
            loader: self,
            prefix: None,
        }
    }
}

pub struct ParameterTree<'loader, C: DeviceContext> {
    loader: &'loader ParameterLoader<'loader, 'loader, C>,
    prefix: Option<String>,
}

impl<'loader, C: DeviceContext> ParameterTree<'loader, C> {
    pub fn new(loader: &'loader ParameterLoader<'loader, 'loader, C>) -> Self {
        Self {
            loader,
            prefix: None,
        }
    }

    pub fn path_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    fn join_prefix(
        &self,
        name: &str,
    ) -> String {
        self.prefix
            .as_ref()
            .map_or_else(|| name.to_string(), |p| format!("{p}.{name}"))
    }

    pub fn subtree(
        &self,
        name: &str,
    ) -> Result<Self, ParameterLoaderError> {
        let new_prefix = self.join_prefix(name);
        let num_suffixes = self
            .loader
            .keys()
            .filter_map(|suffix| suffix.strip_prefix(&new_prefix))
            .count();
        if num_suffixes > 0 {
            Ok(Self {
                loader: self.loader,
                prefix: Some(new_prefix),
            })
        } else {
            Err(ParameterLoaderError::SubtreeNotFound(name.to_string()))
        }
    }

    pub fn leaf(
        &self,
        name: &str,
    ) -> Result<C::DeviceArray, ParameterLoaderError> {
        self.loader.get(&self.join_prefix(name))
    }

    pub fn read_extract_at(
        &self,
        name: &str,
        buf: &mut [u8],
        shape: &mut Box<[usize]>,
        data_type: &mut DataType,
    ) -> Result<(), ParameterLoaderError> {
        self.loader.read_extract_at(
            &self.join_prefix(name),
            buf,
            shape,
            data_type,
        )
    }
}
