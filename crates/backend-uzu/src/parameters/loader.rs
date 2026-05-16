use std::{
    collections::{HashMap, hash_map::Keys},
    fs::File,
};

use thiserror::Error;

use super::safetensors_metadata::{HashMetadata as STMetadata, HeaderLoadingError, read_metadata as read_st_metadata};
use crate::{
    ArrayElement, DataType,
    array::{Array, ArrayContextExt},
    backends::common::{Allocation, AllocationType, AsBufferRangeRef, Backend, Context, DenseBuffer},
    utils::fs::file_read_exact_at,
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
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[usize]>,
        data_type: DataType,
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
    },
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

    fn get_leaf<'leaf>(
        &'leaf self,
        key: &str,
    ) -> Result<ParameterLeaf<'file, 'context, 'leaf, C>, ParameterLoaderError<C::Backend>> {
        Ok(ParameterLeaf {
            metadata: self.index.get(key).ok_or_else(|| ParameterLoaderError::KeyNotFound(key.to_string()))?,
            loader: self,
        })
    }

    fn get(
        &self,
        key: &str,
    ) -> Result<Array<C::Backend>, ParameterLoaderError<C::Backend>> {
        let metadata_entry = self.index.get(key).ok_or(ParameterLoaderError::KeyNotFound(key.to_string()))?;
        let (offset, size) = (metadata_entry.offset, metadata_entry.size);
        let mut array = self.context.create_array_uninitialized(&metadata_entry.shape, metadata_entry.data_type);
        if array.size() != size {
            return Err(ParameterLoaderError::SizeMismatch {
                data_type: metadata_entry.data_type,
                shape: metadata_entry.shape.to_owned(),
                expected_size: array.size(),
                actual_size: size,
            });
        }

        file_read_exact_at(self.file, array.as_bytes_mut(), offset as u64)?;
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
        file_read_exact_at(self.file, buf, metadata_entry.offset as u64)?;
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

    pub fn validate_shape(
        &self,
        expected_shape: &[usize],
        expected_data_type: DataType,
    ) -> Result<(), ParameterLoaderError<C::Backend>> {
        let shape = self.shape();
        let data_type = self.data_type();
        if (shape, data_type) != (expected_shape, expected_data_type) {
            return Err(ParameterLoaderError::InvalidTensor {
                shape: shape.into(),
                data_type,
                expected_shape: expected_shape.into(),
                expected_data_type,
            });
        }
        Ok(())
    }

    pub fn read_slice<T: ArrayElement>(&self) -> Result<Box<[T]>, ParameterLoaderError<C::Backend>> {
        let element_count = self.metadata.size / std::mem::size_of::<T>();
        let mut data = vec![T::zeroed(); element_count];
        file_read_exact_at(self.loader.file, bytemuck::cast_slice_mut(&mut data), self.metadata.offset as u64)?;
        Ok(data.into_boxed_slice())
    }

    pub fn read_allocation(&self) -> Result<Allocation<C::Backend>, ParameterLoaderError<C::Backend>> {
        let allocation = self
            .loader
            .context
            .create_allocation(self.metadata.size, AllocationType::Global)
            .map_err(ParameterLoaderError::BackendError)?;
        let buffer_range = allocation.as_buffer_range_ref();
        let range = buffer_range.range();
        file_read_exact_at(
            self.loader.file,
            unsafe {
                std::slice::from_raw_parts_mut(
                    (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start),
                    range.len(),
                )
            },
            self.metadata.offset as u64,
        )?;
        Ok(allocation)
    }
}

pub struct ParameterTree<'loader, C: Context> {
    loader: &'loader ParameterLoader<'loader, 'loader, C>,
    prefix: Option<String>,
}

impl<'loader, C: Context> ParameterTree<'loader, C> {
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

    pub fn leaf_allocation(
        &self,
        name: &str,
    ) -> Result<Allocation<C::Backend>, ParameterLoaderError<C::Backend>> {
        self.leaf(name)?.read_allocation()
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
