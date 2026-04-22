use std::{cell::RefCell, ops::Range, rc::Rc};

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{Backend, Buffer, Context, Encoder, SparseBuffer},
};

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
    ) -> Self {
        let size = size_for_shape(shape, data_type);
        let mut buffer = context.create_sparse_buffer(size).expect("Failed to create sparse buffer");
        // TODO: remove
        buffer.extend(size);
        Self {
            buffer: Rc::new(RefCell::new(buffer)),
            data_type,
            shape: shape.into(),
        }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
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
        let array: SparseArray<Self::Backend> = SparseArray::new(self, data_type, shape);
        let sparse_buffer_rc = array.sparse_buffer();
        let mut sparse_buffer_ref_mut = sparse_buffer_rc.borrow_mut();
        let sparse_buffer = sparse_buffer_ref_mut.buffer_mut();
        sparse_buffer.set_label(Some(label));
        array
    }
}
