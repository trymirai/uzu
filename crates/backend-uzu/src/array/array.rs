use std::fmt;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, BufferRangeMut, BufferRangeRef,
    },
};

pub struct Array<B: Backend, BufferRange: AsBufferRangeMut<Buffer: Buffer<Backend = B>> = Allocation<B>> {
    pub(super) buffer_range: BufferRange,
    pub(super) offset: usize,
    pub(super) shape: Box<[usize]>,
    pub(super) data_type: DataType,
}

impl<B: Backend, BufferRange: AsBufferRangeMut<Buffer: Buffer<Backend = B>>> Array<B, BufferRange> {
    pub unsafe fn from_parts(
        buffer_range: BufferRange,
        offset: usize,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        let required_bytes = size_for_shape(shape, data_type);
        let buffer_range_size = buffer_range.as_buffer_range_ref().range().len();

        assert!(
            offset + required_bytes <= buffer_range_size,
            "Shape {:?} with data type {:?} at offset {} requires {} bytes total, but buffer length is {} bytes",
            shape,
            data_type,
            offset,
            offset + required_bytes,
            buffer_range_size
        );
        Self {
            buffer_range,
            offset,
            shape: shape.into(),
            data_type,
        }
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

    pub fn as_buffer_range_ref<'a>(&'a self) -> BufferRangeRef<'a, BufferRange::Buffer> {
        self.buffer_range.as_buffer_range_ref().subrange(self.offset..self.offset + self.size())
    }

    pub fn as_buffer_range_mut<'a>(&'a mut self) -> BufferRangeMut<'a, BufferRange::Buffer> {
        self.buffer_range.as_buffer_range_mut().subrange(self.offset..self.offset + self.size())
    }
}

impl<B: Backend> Array<B, Allocation<B>> {
    pub unsafe fn from_allocation(
        allocation: Allocation<B>,
        offset: usize,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        unsafe { Self::from_parts(allocation, offset, shape, data_type) }
    }

    pub fn allocation(&self) -> &Allocation<B> {
        &self.buffer_range
    }

    pub fn allocation_mut(&mut self) -> &mut Allocation<B> {
        &mut self.buffer_range
    }

    pub fn into_allocation(self) -> Allocation<B> {
        assert_eq!(self.offset, 0, "Array view cannot be converted into Allocation");
        assert_eq!(
            size_for_shape(&self.shape, self.data_type),
            self.buffer_range.as_buffer_range_ref().range().len(),
            "Partial Array view cannot be converted into Allocation",
        );
        self.buffer_range
    }
}

impl<B: Backend, BufferRange: AsBufferRangeMut<Buffer: Buffer<Backend = B>>> fmt::Debug for Array<B, BufferRange> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter
            .debug_struct("Array")
            .field("offset", &self.offset)
            .field("shape", &self.shape)
            .field("data_type", &self.data_type)
            .finish()
    }
}
