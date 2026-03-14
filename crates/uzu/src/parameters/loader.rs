use std::{
    collections::{HashMap, hash_map::Keys},
    fs::File,
    os::unix::fs::FileExt,
};

use half::{bf16, f16};
use thiserror::Error;

use super::safetensors_metadata::{HashMetadata as STMetadata, HeaderLoadingError, read_metadata as read_st_metadata};
use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::{Backend, Buffer, Context},
};

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
pub enum ParameterLoaderError<B: Backend> {
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
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Failed to read data")]
    ArrayLoadingError(#[from] std::io::Error),
    #[error("Unsupported host tensor dtype: {0:?}")]
    UnsupportedHostTensorDataType(DataType),
}

pub struct ParameterLoader<'context, 'file, C: Context>
where
    'file: 'context,
{
    context: &'context C,
    index: HashMap<String, ParameterMetadata>,
    file: &'file File,
}

impl<'file, 'context, C: Context> ParameterLoader<'file, 'context, C>
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

    pub fn get_leaf<'leaf>(
        &'leaf self,
        key: &str,
    ) -> Result<ParameterLeaf<'file, 'context, 'leaf, C>, ParameterLoaderError<C::Backend>> {
        Ok(ParameterLeaf {
            key: key.to_string(),
            metadata: self.index.get(key).ok_or_else(|| ParameterLoaderError::KeyNotFound(key.to_string()))?,
            loader: self,
        })
    }

    pub fn get(
        &self,
        key: &str,
    ) -> Result<Array<C::Backend>, ParameterLoaderError<C::Backend>> {
        let metadata_entry = self.index.get(key).ok_or(ParameterLoaderError::KeyNotFound(key.to_string()))?;
        let (offset, size) = (metadata_entry.offset, metadata_entry.size);
        let array_key = key.replace(".", "_");
        let array_label = format!("parameter_loader_{array_key}");
        let mut array = self.context.create_array(&metadata_entry.shape, metadata_entry.data_type, &array_label);
        let expected_size = array.size();
        if expected_size != size {
            return Err(ParameterLoaderError::SizeMismatch {
                data_type: metadata_entry.data_type,
                shape: metadata_entry.shape.to_owned(),
                expected_size,
                actual_size: size,
            });
        }
        self.file.read_exact_at(array.as_bytes_mut(), offset as u64)?;
        Ok(array)
    }

    pub fn read_extract_at(
        &self,
        key: &str,
        buf: &mut [u8],
        shape: &mut Box<[usize]>,
        data_type: &mut DataType,
    ) -> Result<(), ParameterLoaderError<C::Backend>> {
        let metadata_entry = self.index.get(key).ok_or(ParameterLoaderError::KeyNotFound(key.to_string()))?;
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

pub struct ParameterLeaf<'file, 'context, 'leaf, C: Context> {
    key: String,
    metadata: &'leaf ParameterMetadata,
    loader: &'leaf ParameterLoader<'context, 'file, C>,
}

impl<'file, 'context, 'leaf, C: Context> ParameterLeaf<'file, 'context, 'leaf, C> {
    pub fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    pub fn data_type(&self) -> DataType {
        self.metadata.data_type
    }

    pub fn size(&self) -> usize {
        self.metadata.size
    }

    pub fn read_buffer(&self) -> Result<<C::Backend as Backend>::Buffer, ParameterLoaderError<C::Backend>> {
        let mut buffer =
            self.loader.context.create_buffer(self.metadata.size).map_err(ParameterLoaderError::BackendError)?;
        buffer.set_label(Some(&format!("parameter_loader_{}", self.key.replace(".", "_"))));
        self.loader.file.read_exact_at(
            unsafe { std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr() as *mut u8, self.metadata.size) },
            self.metadata.offset as u64,
        )?;
        Ok(buffer)
    }

    pub fn read_f32_tensor(&self) -> Result<(Box<[usize]>, Vec<f32>), ParameterLoaderError<C::Backend>> {
        let num_elements = self
            .metadata
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| ParameterLoaderError::SizeMismatch {
                data_type: self.metadata.data_type,
                shape: self.metadata.shape.to_owned(),
                expected_size: usize::MAX,
                actual_size: self.metadata.size,
            })?;
        let expected_size = num_elements
            .checked_mul(self.metadata.data_type.size_in_bytes())
            .ok_or_else(|| ParameterLoaderError::SizeMismatch {
                data_type: self.metadata.data_type,
                shape: self.metadata.shape.to_owned(),
                expected_size: usize::MAX,
                actual_size: self.metadata.size,
            })?;
        if expected_size != self.metadata.size {
            return Err(ParameterLoaderError::SizeMismatch {
                data_type: self.metadata.data_type,
                shape: self.metadata.shape.to_owned(),
                expected_size,
                actual_size: self.metadata.size,
            });
        }

        let mut bytes = vec![0u8; self.metadata.size];
        self.loader.file.read_exact_at(&mut bytes, self.metadata.offset as u64)?;
        let values = match self.metadata.data_type {
            DataType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            DataType::F16 => bytes
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect(),
            DataType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect(),
            other => return Err(ParameterLoaderError::UnsupportedHostTensorDataType(other)),
        };
        Ok((self.metadata.shape.to_owned(), values))
    }
}

pub struct ParameterTree<'loader, C: Context> {
    loader: &'loader ParameterLoader<'loader, 'loader, C>,
    prefix: Option<String>,
}

impl<'loader, C: Context> ParameterTree<'loader, C> {
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
        self.prefix.as_ref().map_or_else(|| name.to_string(), |p| format!("{p}.{name}"))
    }

    pub fn subtree(
        &self,
        name: &str,
    ) -> Result<Self, ParameterLoaderError<C::Backend>> {
        let new_prefix = self.join_prefix(name);
        let num_suffixes = self.loader.keys().filter_map(|suffix| suffix.strip_prefix(&new_prefix)).count();
        if num_suffixes > 0 {
            Ok(Self {
                loader: self.loader,
                prefix: Some(new_prefix),
            })
        } else {
            Err(ParameterLoaderError::SubtreeNotFound(name.to_string()))
        }
    }

    pub fn leaf_array(
        &self,
        name: &str,
    ) -> Result<Array<C::Backend>, ParameterLoaderError<C::Backend>> {
        self.loader.get(&self.join_prefix(name))
    }

    pub fn leaf<'leaf>(
        &'leaf self,
        name: &str,
    ) -> Result<ParameterLeaf<'loader, 'loader, 'leaf, C>, ParameterLoaderError<C::Backend>> {
        self.loader.get_leaf(&self.join_prefix(name))
    }

    pub fn read_extract_at(
        &self,
        name: &str,
        buf: &mut [u8],
        shape: &mut Box<[usize]>,
        data_type: &mut DataType,
    ) -> Result<(), ParameterLoaderError<C::Backend>> {
        self.loader.read_extract_at(&self.join_prefix(name), buf, shape, data_type)
    }
}

pub fn resolve_subtree<'tree, C: Context>(
    tree: &'tree ParameterTree<C>,
    candidates: &[&str],
) -> ParameterTree<'tree, C> {
    for candidate in candidates {
        if let Ok(subtree) = tree.subtree(candidate) {
            return subtree;
        }
    }
    panic!("Could not find any of {:?} in parameter tree", candidates);
}
