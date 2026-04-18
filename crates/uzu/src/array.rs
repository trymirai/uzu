use std::{fmt, ops::Range, os::raw::c_void, ptr::NonNull};

use ndarray::{ArrayView, Dimension, IxDyn};

use crate::{
    ArrayElement, DataType,
    backends::common::{Allocation, AllocationType, Backend, Buffer, Context},
};

pub struct Array<B: Backend> {
    allocation: Allocation<B>,
    offset: usize,
    shape: Box<[usize]>,
    data_type: DataType,
}

impl<B: Backend> Array<B> {
    // Constructors
    pub unsafe fn from_allocation(
        allocation: Allocation<B>,
        offset: usize,
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        let required_bytes = size_for_shape(shape, data_type);
        let allocation_len = allocation.as_buffer_range().1.len();
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
            allocation: allocation.view_at_offset(offset, required_bytes),
            offset,
            shape: shape.into(),
            data_type,
        }
    }

    pub fn view(
        &self,
        shape: &[usize],
    ) -> Self {
        self.view_at_offset(0, shape)
    }

    pub fn view_at_offset(
        &self,
        byte_offset: usize,
        shape: &[usize],
    ) -> Self {
        let required_bytes = size_for_shape(shape, self.data_type);
        let offset = self.offset + byte_offset;
        let allocation_len = self.allocation.as_buffer_range().1.len();
        assert!(
            byte_offset + required_bytes <= allocation_len,
            "Shape {:?} with data type {:?} at offset {} requires {} bytes total, but allocation length is {} bytes",
            shape,
            self.data_type,
            offset,
            offset + required_bytes,
            allocation_len
        );
        Self {
            allocation: self.allocation.view_at_offset(byte_offset, required_bytes),
            offset,
            shape: shape.into(),
            data_type: self.data_type,
        }
    }

    // Getters
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
        &self.allocation
    }

    pub fn into_allocation(self) -> Allocation<B> {
        self.allocation
    }

    pub fn as_buffer_range(&self) -> (&B::Buffer, Range<usize>) {
        self.allocation.as_buffer_range()
    }

    // Utility
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        let (buffer, range) = self.as_buffer_range();
        unsafe { buffer.cpu_ptr().add(range.start) }
    }

    pub fn size(&self) -> usize {
        size_for_shape(&self.shape, self.data_type)
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    fn validate_element_type<T: ArrayElement>(&self) {
        assert_eq!(
            T::data_type(),
            self.data_type,
            "Invalid data type, expected {:?}, actual {:?}",
            T::data_type(),
            self.data_type
        );
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.cpu_ptr().as_ptr() as *const u8, self.size()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.cpu_ptr().as_ptr() as *mut u8, self.size()) }
    }

    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.validate_element_type::<T>();
        bytemuck::cast_slice(self.as_bytes())
    }

    pub fn as_slice_mut<T: ArrayElement>(&mut self) -> &mut [T] {
        self.validate_element_type::<T>();
        bytemuck::cast_slice_mut(self.as_bytes_mut())
    }

    pub fn as_view<T: ArrayElement>(&self) -> ArrayView<'_, T, IxDyn> {
        ArrayView::from_shape(IxDyn(self.shape()), self.as_slice::<T>()).expect("Failed to create array view")
    }

    pub fn copy_from_array<C: Backend>(
        &mut self,
        other: &Array<C>,
    ) {
        assert_eq!(self.shape, other.shape);
        assert_eq!(self.data_type, other.data_type);

        self.as_bytes_mut().copy_from_slice(other.as_bytes());
    }

    pub fn copy_slice<C: Backend>(
        &mut self,
        source: &Array<C>,
        axis: usize,
        src_range: Range<usize>,
        dst_offset: usize,
    ) {
        assert_eq!(self.shape().len(), source.shape().len(), "Rank mismatch");
        assert!(axis < self.shape().len(), "Axis out of bounds");
        for (i, (a, b)) in self.shape().iter().zip(source.shape()).enumerate() {
            if i != axis {
                assert_eq!(a, b, "Shapes must match on all non-sliced axes");
            }
        }
        assert_eq!(self.data_type(), source.data_type(), "Arrays must have the same data type");

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
        let src_buf = source.as_bytes();
        let dst_buf = self.as_bytes_mut();
        let copy_bytes = rows_to_copy * block_size_bytes;

        for i in 0..num_blocks {
            let src_block_start = i * src_stride_bytes;
            let dst_block_start = i * dst_stride_bytes;

            let src_start = src_block_start + src_range.start * block_size_bytes;
            let dst_start = dst_block_start + dst_offset * block_size_bytes;

            let src_slice = &src_buf[src_start..src_start + copy_bytes];
            let dst_slice = &mut dst_buf[dst_start..dst_start + copy_bytes];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    pub fn copy_from_view<T: ArrayElement, D: Dimension>(
        &mut self,
        view: ArrayView<T, D>,
    ) {
        assert_eq!(self.data_type(), T::data_type());

        let dst_slice = self.as_slice_mut::<T>();

        if let Some(src_slice) = view.as_slice_memory_order() {
            assert!(src_slice.len() <= dst_slice.len());
            dst_slice[..src_slice.len()].copy_from_slice(src_slice);
        } else {
            assert!(view.len() <= dst_slice.len());
            for (d, s) in dst_slice.iter_mut().zip(view.iter()) {
                *d = *s;
            }
        }
    }
}

pub fn allocation_as_slice<T: ArrayElement, B: Backend>(allocation: &Allocation<B>) -> &[T] {
    let (buffer, range) = allocation.as_buffer_range();
    unsafe {
        let bytes = std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start), range.len());
        bytemuck::cast_slice(bytes)
    }
}

impl<B: Backend> Clone for Array<B> {
    fn clone(&self) -> Self {
        Self {
            allocation: self.allocation.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
            data_type: self.data_type,
        }
    }
}

impl<B: Backend> fmt::Debug for Array<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Array")
            .field("offset", &self.offset)
            .field("shape", &self.shape)
            .field("data_type", &self.data_type)
            .finish()
    }
}

pub fn size_for_shape(
    shape: &[usize],
    data_type: DataType,
) -> usize {
    let Some(last_dim) = shape.last() else {
        return data_type.size_in_bytes();
    };

    let bits_per_row = last_dim * data_type.size_in_bits();
    let padded_bytes_per_row = bits_per_row.div_ceil(8);

    let num_rows: usize = shape.iter().rev().skip(1).product();

    num_rows * padded_bytes_per_row
}

// Array extends Context with helper functions to create arrays from context

pub trait ArrayContextExt {
    type Backend: Backend;

    fn create_array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> Array<Self::Backend>;

    fn create_array_zeros(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> Array<Self::Backend> {
        let mut array = self.create_array_uninitialized(shape, data_type, label);
        array.as_bytes_mut().fill(0);
        array
    }

    fn create_array_from<T: ArrayElement>(
        &self,
        shape: &[usize],
        data: &[T],
        label: &str,
    ) -> Array<Self::Backend> {
        let size_from_shape: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size_from_shape,
            "Shape size {} and data size {} are not equal",
            size_from_shape,
            data.len()
        );

        let mut array = self.create_array_uninitialized(shape, T::data_type(), label);
        array.as_slice_mut().copy_from_slice(data);
        array
    }
}

impl<C: Context> ArrayContextExt for C {
    type Backend = C::Backend;

    fn create_array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> Array<Self::Backend> {
        let _ = label;
        let allocation = self
            .create_allocation(size_for_shape(shape, data_type), AllocationType::Global)
            .expect("Failed to create allocation");
        unsafe { Array::from_allocation(allocation, 0, shape, data_type) }
    }
}
