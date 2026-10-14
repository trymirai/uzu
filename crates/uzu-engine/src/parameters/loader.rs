use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
};

use super::{
    ParameterReadSource,
    error::ParameterLoaderError,
    safetensors_metadata::{
        HashMetadata, HeaderLoadingError, read_exact_at, read_metadata as read_st_metadata, read_metadata_from_bytes,
        read_metadata_from_source,
    },
};
use crate::{
    array::{ArrayElement, size_for_shape},
    backends::common::{Backend, BufferMut, Context},
    data_type::DataType,
    utils::strict_serde::DeserializeStrictOwned,
};

struct ParameterMetadata {
    shape: Box<[u32]>,
    data_type: DataType,
    offset: u64,
    size: usize,
}

pub struct ParameterLoader<'a, B: Backend> {
    context: &'a B::Context,
    index: HashMap<String, ParameterMetadata>,
    metadata: HashMap<String, String>,
    validated_tensors: RefCell<HashSet<String>>,
    source: ParameterSource<'a>,
    buffer_initializer: Option<&'a ParameterBufferInitializer<'a, B>>,
}

/// Optional initialization of a fresh backend buffer. The default File/bytes
/// loaders read directly into CPU-accessible storage; streaming GPU backends
/// can initialize without first exposing a persistent CPU mirror.
pub type ParameterBufferInitializer<'a, B> =
    dyn Fn(&<B as Backend>::Context, u64, usize) -> Result<<B as Backend>::GlobalBuffer, ParameterLoaderError<B>> + 'a;

enum ParameterSource<'a> {
    File(&'a File),
    Bytes(&'a [u8]),
    Reader(&'a dyn ParameterReadSource),
}

impl ParameterSource<'_> {
    fn read(
        &self,
        destination: &mut [u8],
        offset: u64,
    ) -> std::io::Result<()> {
        match self {
            Self::File(file) => read_exact_at(file, destination, offset),
            Self::Reader(source) => source.read_exact_at(destination, offset),
            Self::Bytes(bytes) => {
                let offset = usize::try_from(offset)
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "tensor offset overflow"))?;
                let end = offset
                    .checked_add(destination.len())
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "tensor offset overflow"))?;
                let source = bytes
                    .get(offset..end)
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated tensor"))?;
                destination.copy_from_slice(source);
                Ok(())
            },
        }
    }
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub fn new(
        file: &'a File,
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let (global_offset, st_metadata) = read_st_metadata(file)?;
        Self::with_metadata(
            context,
            ParameterSource::File(file),
            global_offset,
            st_metadata,
            file.metadata()?.len(),
            None,
        )
    }

    pub fn from_bytes(
        bytes: &'a [u8],
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let (global_offset, st_metadata) = read_metadata_from_bytes(bytes)?;
        Self::with_metadata(
            context,
            ParameterSource::Bytes(bytes),
            global_offset,
            st_metadata,
            bytes.len() as u64,
            None,
        )
    }

    pub fn from_source(
        source: &'a dyn ParameterReadSource,
        context: &'a B::Context,
        buffer_initializer: Option<&'a ParameterBufferInitializer<'a, B>>,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let (global_offset, metadata) = read_metadata_from_source(source)?;
        Self::with_metadata(
            context,
            ParameterSource::Reader(source),
            global_offset,
            metadata,
            source.len(),
            buffer_initializer,
        )
    }

    fn with_metadata(
        context: &'a B::Context,
        source: ParameterSource<'a>,
        global_offset: u64,
        st_metadata: HashMetadata,
        source_size: u64,
        buffer_initializer: Option<&'a ParameterBufferInitializer<'a, B>>,
    ) -> Result<Self, ParameterLoaderError<B>> {
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
                let end = offset.checked_add(size).ok_or_else(|| HeaderLoadingError::TensorOffsetOverflow {
                    key: key.clone().into_boxed_str(),
                    global_offset: offset,
                    local_begin: size,
                })?;
                if end > source_size {
                    return Err(HeaderLoadingError::TensorOutOfBounds {
                        key: key.into_boxed_str(),
                        end,
                        available: source_size,
                    });
                }
                let size = usize::try_from(size).map_err(|_| HeaderLoadingError::TensorTooLarge {
                    key: key.clone().into_boxed_str(),
                    size,
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
            source,
            buffer_initializer,
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
        expected_shape: &[u32],
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
    pub fn read_slice<T: ArrayElement>(&self) -> Result<Box<[T]>, ParameterLoaderError<B>> {
        let element_count = self.metadata.size / std::mem::size_of::<T>();
        let mut data = vec![T::zeroed(); element_count];
        let destination = bytemuck::cast_slice_mut(&mut data);
        self.loader.source.read(destination, self.metadata.offset)?;
        Ok(data.into_boxed_slice())
    }

    pub fn read_buffer(&self) -> Result<B::GlobalBuffer, ParameterLoaderError<B>> {
        if let Some(initialize) = self.loader.buffer_initializer {
            return initialize(self.loader.context, self.metadata.offset, self.metadata.size);
        }
        let mut buffer =
            self.loader.context.create_buffer(self.metadata.size).map_err(ParameterLoaderError::BackendError)?;
        self.loader.source.read(buffer.as_slice_mut::<u8>(), self.metadata.offset)?;
        Ok(buffer)
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
    ) -> Self {
        Self {
            loader: self.loader,
            prefix: Some(self.join_prefix(name)),
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
