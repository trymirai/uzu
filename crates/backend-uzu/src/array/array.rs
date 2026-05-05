use std::{cell::RefCell, rc::Rc};

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{Backend, Buffer},
};

#[derive(Debug)]
pub struct Array<B: Backend, Buf: Buffer<Backend = B> = <B as Backend>::DenseBuffer> {
    pub(super) buffer: Rc<RefCell<Buf>>,
    pub(super) offset: usize,
    pub(super) shape: Box<[usize]>,
    pub(super) data_type: DataType,
}

impl<B: Backend, Buf: Buffer<Backend = B>> Array<B, Buf> {
    // Constructors
    pub unsafe fn from_parts(
        buffer: Rc<RefCell<Buf>>,
        offset: usize,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        let required_bytes = size_for_shape(shape, data_type);
        assert!(
            offset + required_bytes <= buffer.borrow().size(),
            "Shape {:?} with data type {:?} at offset {} requires {} bytes total, but buffer length is {} bytes",
            shape,
            data_type,
            offset,
            offset + required_bytes,
            buffer.borrow().size()
        );
        Self {
            buffer: buffer.clone(),
            offset,
            shape: shape.into(),
            data_type,
        }
    }

    pub fn view(
        &self,
        shape: &[usize],
    ) -> Self {
        unsafe { Self::from_parts(self.buffer.clone(), self.offset, shape, self.data_type) }
    }

    // Getters
    pub fn buffer(&self) -> Rc<RefCell<Buf>> {
        self.buffer.clone()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn size(&self) -> usize {
        size_for_shape(&self.shape, self.data_type)
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<B: Backend, Buf: Buffer<Backend = B>> Clone for Array<B, Buf> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
            data_type: self.data_type,
        }
    }
}
