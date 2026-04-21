use std::{cell::RefCell, ops::Range, rc::Rc};

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{Backend, Buffer, Context, SparseBuffer},
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

        todo!()
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
        array.sparse_buffer().borrow_mut().buffer().borrow_mut().set_label(Some(label));
        array
    }
}
