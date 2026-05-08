use std::fmt;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{Allocation, AsBufferRangeRef, Backend},
};

pub struct Array<B: Backend> {
    pub(super) allocation: Option<Allocation<B>>,
    pub(super) offset: usize,
    pub(super) shape: Box<[usize]>,
    pub(super) data_type: DataType,
}

impl<B: Backend> Array<B> {
    fn assert_offset_alignment(
        offset: usize,
        data_type: DataType,
    ) {
        let alignment = data_type.size_in_bytes();
        assert!(offset % alignment == 0, "Array offset {} is not aligned to element size {}", offset, alignment);
    }

    pub unsafe fn from_allocation(
        allocation: Allocation<B>,
        offset: usize,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        let required_bytes = size_for_shape(shape, data_type);
        let allocation_len = allocation.as_buffer_range_ref().range().len();
        Self::assert_offset_alignment(offset, data_type);
        assert!(
            offset + required_bytes <= allocation_len,
            "Shape {:?} with data type {:?} at offset {} requires {} bytes total, but allocation length is {} bytes",
            shape,
            data_type,
            offset,
            offset + required_bytes,
            allocation_len
        );
        Self {
            allocation: Some(allocation),
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

    pub fn allocation(&self) -> &Allocation<B> {
        self.allocation.as_ref().expect("Empty Array has no backing allocation")
    }

    pub fn allocation_mut(&mut self) -> &mut Allocation<B> {
        self.allocation.as_mut().expect("Empty Array has no backing allocation")
    }

    pub fn into_allocation(self) -> Allocation<B> {
        assert_eq!(self.offset, 0, "Array view cannot be converted into Allocation");
        let allocation = self.allocation.expect("Empty Array has no backing allocation");
        assert_eq!(
            size_for_shape(&self.shape, self.data_type),
            allocation.as_buffer_range_ref().range().len(),
            "Partial Array view cannot be converted into Allocation",
        );
        allocation
    }

    pub fn size(&self) -> usize {
        size_for_shape(&self.shape, self.data_type)
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<B: Backend> fmt::Debug for Array<B> {
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
