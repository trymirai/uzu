use std::{cell::RefCell, ops::Range, rc::Rc};

use backend_uzu::ArrayElement;
use thiserror::Error;

use crate::{
    Array, ArrayContextExt, DataType,
    array::size_for_shape,
    backends::common::{Backend, Buffer, Context, Encoder, SparseBuffer},
};

#[derive(Debug, Error)]
pub enum SparseArrayError<B: Backend> {
    #[error("Failed to create SparseBuffer: {0}")]
    CreateBufferError(B::Error),
    #[error("Failed to create Encoder: {0}")]
    CreateEncoderError(B::Error),
}

#[derive(Debug)]
pub struct SparseArray<B: Backend> {
    buffer: Rc<RefCell<B::SparseBuffer>>,
    data_type: DataType,
    shape: Box<[usize]>,
}

impl<B: Backend> SparseArray<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        shape: &[usize],
    ) -> Result<Self, SparseArrayError<B>> {
        let size = size_for_shape(shape, data_type);
        let buffer = context.create_sparse_buffer(size).map_err(|e| SparseArrayError::CreateBufferError(e))?;
        Ok(Self {
            buffer: Rc::new(RefCell::new(buffer)),
            data_type,
            shape: shape.into(),
        })
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Bytes needed for `rows` leading entries along axis 0.
    fn row_bytes(&self) -> usize {
        self.shape.iter().skip(1).product::<usize>() * self.data_type.size_in_bytes()
    }

    /// Ensure the underlying sparse buffer is backed for at least `rows`
    /// entries along axis 0. Grows on demand; never shrinks.
    pub fn ensure_rows(
        &self,
        rows: usize,
    ) {
        let required = rows * self.row_bytes();
        let mut buffer = self.buffer.borrow_mut();
        let current = buffer.length();
        if current < required {
            buffer.extend(required - current);
        }
    }

    /// Ensure the buffer is fully backed up to its logical shape.
    /// Needed when writes happen on a non-leading axis.
    pub fn ensure_full(&self) {
        let required = size_for_shape(&self.shape, self.data_type);
        let mut buffer = self.buffer.borrow_mut();
        let current = buffer.length();
        if current < required {
            buffer.extend(required - current);
        }
    }

    pub fn copy_slice(
        &mut self,
        source: &SparseArray<B>,
        axis: usize,
        src_range: Range<usize>,
        dst_offset: usize,
        encoder: &mut Encoder<B>,
    ) {
        assert_eq!(self.data_type(), source.data_type(), "Arrays must have the same data type");
        assert_eq!(self.shape().len(), source.shape().len(), "Rank mismatch");
        assert!(axis < self.shape().len(), "Axis out of bounds");
        for i in 0..source.shape().len() {
            if i != axis {
                let src_size = source.shape()[i];
                let dst_size = self.shape()[i];
                assert_eq!(src_size, dst_size, "Shapes must match on all non-sliced axes");
            }
        }

        let elem_size = self.data_type().size_in_bytes();

        // The number of contiguous elements to copy for each slice operation
        let block_size_elems = self.shape().iter().skip(axis + 1).product::<usize>();
        let block_size_bytes = block_size_elems * elem_size;

        // The total number of blocks to copy
        let num_blocks: usize = self.shape().iter().take(axis).product();

        // Strides between the start of each block
        let src_stride_bytes = source.shape()[axis] * block_size_bytes;
        let dst_stride_bytes = self.shape()[axis] * block_size_bytes;

        let rows_to_copy = src_range.end - src_range.start;
        assert!(dst_offset + rows_to_copy <= self.shape()[axis]);
        assert!(src_range.end <= source.shape()[axis]);

        // On-demand extend: make sure both source and destination sparse buffers
        // are backed for the byte ranges about to be read/written.
        if axis == 0 {
            source.ensure_rows(src_range.end);
            self.ensure_rows(dst_offset + rows_to_copy);
        } else {
            source.ensure_full();
            self.ensure_full();
        }

        let src_sparse_buffer_rc = source.sparse_buffer();
        let src_sparse_buffer = src_sparse_buffer_rc.borrow();
        let dst_sparse_buffer_rc = self.sparse_buffer();
        let mut dst_sparse_buffer = dst_sparse_buffer_rc.borrow_mut();

        let copy_bytes = rows_to_copy * block_size_bytes;
        for i in 0..num_blocks {
            let src_block_start = i * src_stride_bytes;
            let dst_block_start = i * dst_stride_bytes;

            let src_start = src_block_start + src_range.start * block_size_bytes;
            let src_rng = Range {
                start: src_start,
                end: src_start + copy_bytes,
            };

            let dst_start = dst_block_start + dst_offset * block_size_bytes;
            let dst_rng = Range {
                start: dst_start,
                end: dst_start + copy_bytes,
            };

            encoder.encode_copy(src_sparse_buffer.buffer(), src_rng, dst_sparse_buffer.buffer_mut(), dst_rng);
        }
    }

    pub fn sparse_buffer(&self) -> Rc<RefCell<B::SparseBuffer>> {
        self.buffer.clone()
    }

    pub fn to_array(
        &self,
        context: &B::Context,
    ) -> Result<Array<B>, SparseArrayError<B>> {
        let sparse_buffer = self.buffer.borrow();
        let src_buffer = sparse_buffer.buffer();
        let logical_length = size_for_shape(&self.shape(), self.data_type);
        // Only copy the bytes that are actually mapped in the sparse buffer.
        // Reading past `sparse_buffer.length()` can hit unmapped pages.
        let length = logical_length.min(sparse_buffer.length());

        let array = context.create_array_uninitialized(self.shape(), self.data_type, "");
        let array_buffer = array.buffer();
        let mut dst_buffer = array_buffer.borrow_mut();

        if length > 0 {
            let mut encoder = Encoder::<B>::new(context).map_err(|err| SparseArrayError::CreateEncoderError(err))?;
            encoder.encode_copy(&src_buffer, 0..length, &mut dst_buffer, 0..length);
            encoder
                .end_encoding()
                .submit()
                .wait_until_completed()
                .map_err(|err| SparseArrayError::CreateEncoderError(err))?;
        }

        Ok(array)
    }

    pub fn read_bytes(
        &self,
        context: &B::Context,
        range: Range<usize>,
    ) -> Result<Vec<u8>, SparseArrayError<B>> {
        assert!(range.end <= self.sparse_buffer().borrow().length());

        let length = range.len();
        if length == 0 {
            return Ok(Vec::new());
        }

        let mut dst_buffer = context.create_buffer(length).map_err(|err| SparseArrayError::CreateBufferError(err))?;

        let mut encoder = Encoder::<B>::new(context).map_err(|err| SparseArrayError::CreateEncoderError(err))?;
        encoder.encode_copy(self.buffer.borrow().buffer(), range, &mut dst_buffer, 0..length);
        encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| SparseArrayError::CreateEncoderError(err))?;

        let bytes = unsafe { std::slice::from_raw_parts(dst_buffer.cpu_ptr().as_ptr() as *const u8, length) };
        Ok(bytes.to_vec())
    }

    pub fn read_typed<T: ArrayElement>(
        &self,
        context: &B::Context,
    ) -> Result<Vec<T>, SparseArrayError<B>> {
        let data_type_size = T::data_type().size_in_bytes();
        let slice_size = self.sparse_buffer().borrow().length() / data_type_size;
        Ok(self.read_typed_range(context, 0..slice_size)?)
    }

    pub fn read_typed_range<T: ArrayElement>(
        &self,
        context: &B::Context,
        range: Range<usize>,
    ) -> Result<Vec<T>, SparseArrayError<B>> {
        let elem_size = T::data_type().size_in_bytes();
        let bytes_range = Range {
            start: range.start * elem_size,
            end: range.end * elem_size,
        };
        let bytes = self.read_bytes(context, bytes_range)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    pub fn write_bytes(
        &mut self,
        context: &B::Context,
        bytes: &[u8],
        offset: usize,
    ) -> Result<(), SparseArrayError<B>> {
        if bytes.is_empty() {
            return Ok(());
        }

        {
            let mut sparse_buffer = self.buffer.borrow_mut();
            let required = offset + bytes.len();
            let current_length = sparse_buffer.length();
            if current_length < required {
                sparse_buffer.extend(required - current_length);
            }
        }

        let src_buffer = context.create_buffer(bytes.len()).map_err(|err| SparseArrayError::CreateBufferError(err))?;
        let src_slice =
            unsafe { std::slice::from_raw_parts_mut(src_buffer.cpu_ptr().as_ptr() as *mut u8, bytes.len()) };
        src_slice.copy_from_slice(bytes);
        let src_range = Range {
            start: 0,
            end: bytes.len(),
        };

        let sparse_buffer_rc = self.sparse_buffer();
        let mut sparse_buffer = sparse_buffer_rc.borrow_mut();
        let mut dst_buffer = sparse_buffer.buffer_mut();
        let dst_range = Range {
            start: offset,
            end: offset + bytes.len(),
        };

        let mut encoder = Encoder::<B>::new(context).map_err(|err| SparseArrayError::CreateEncoderError(err))?;
        encoder.encode_copy(&src_buffer, src_range, &mut dst_buffer, dst_range);
        encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| SparseArrayError::CreateEncoderError(err))?;
        Ok(())
    }

    pub fn write_typed<T: ArrayElement>(
        &mut self,
        context: &B::Context,
        slice: &[T],
        offset: usize,
    ) -> Result<(), SparseArrayError<B>> {
        let bytes_offset = offset * T::data_type().size_in_bytes();
        let bytes_slice: &[u8] = bytemuck::cast_slice(slice);
        self.write_bytes(context, bytes_slice, bytes_offset)?;
        Ok(())
    }
}

impl<B: Backend> Clone for SparseArray<B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            data_type: self.data_type(),
            shape: self.shape.clone(),
        }
    }
}

pub trait SparseArrayContext {
    type Backend: Backend;

    fn create_sparse_array(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> SparseArray<Self::Backend>;
}

impl<C: Context> SparseArrayContext for C {
    type Backend = C::Backend;

    fn create_sparse_array(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> SparseArray<Self::Backend> {
        let array: SparseArray<Self::Backend> =
            SparseArray::new(self, data_type, shape).expect("Failed to create SparseArray");
        let sparse_buffer_rc = array.sparse_buffer();
        let mut sparse_buffer_ref_mut = sparse_buffer_rc.borrow_mut();
        let sparse_buffer = sparse_buffer_ref_mut.buffer_mut();
        sparse_buffer.set_label(Some(label));
        array
    }
}
