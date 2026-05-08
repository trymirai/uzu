use std::{mem::size_of, ops::Range, os::raw::c_void, ptr::NonNull};

use ndarray::{ArrayView, Dimension, IxDyn};
use thiserror::Error;

use crate::{
    Array, ArrayElement,
    backends::common::{Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, DenseBuffer},
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AllocationAccessError {
    #[error("allocation length {byte_len} is not divisible by element size {element_size}")]
    LengthNotMultiple {
        byte_len: usize,
        element_size: usize,
    },
    #[error("allocation write length {write_len} exceeds allocation length {range_len}")]
    WriteExceedsRange {
        write_len: usize,
        range_len: usize,
    },
}

impl<B: Backend> Array<B> {
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        if self.size() == 0 {
            return NonNull::new(self.data_type().size_in_bytes() as *mut c_void).expect("dtype-aligned empty pointer");
        }
        let buffer_range = self.allocation().as_buffer_range_ref();
        let range = buffer_range.range();
        unsafe { buffer_range.buffer().cpu_ptr().add(range.start + self.offset()) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.cpu_ptr().as_ptr() as *const u8, self.size()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = self.size();
        let offset = self.offset();
        let data_type = self.data_type();
        let Some(allocation) = self.allocation.as_mut() else {
            assert_eq!(size, 0, "Empty Array has no backing allocation");
            let pointer = NonNull::new(data_type.size_in_bytes() as *mut c_void).expect("dtype-aligned empty pointer");
            return unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr() as *mut u8, size) };
        };
        let buffer_range = allocation.as_buffer_range_mut();
        let range = buffer_range.range();
        unsafe {
            std::slice::from_raw_parts_mut(
                (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start + offset),
                size,
            )
        }
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
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.data_type(), other.data_type());

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
        for (dimension_index, (destination_dim, source_dim)) in self.shape().iter().zip(source.shape()).enumerate() {
            if dimension_index != axis {
                assert_eq!(destination_dim, source_dim, "Shapes must match on all non-sliced axes");
            }
        }
        assert_eq!(self.data_type(), source.data_type(), "Arrays must have the same data type");

        let element_size = self.data_type().size_in_bytes();
        let block_size_elements = self.shape().iter().skip(axis + 1).product::<usize>();
        let block_size_bytes = block_size_elements * element_size;
        let num_blocks: usize = self.shape().iter().take(axis).product();
        let source_stride_bytes = source.shape()[axis] * block_size_bytes;
        let destination_stride_bytes = self.shape()[axis] * block_size_bytes;

        let rows_to_copy = src_range.end - src_range.start;
        assert!(dst_offset + rows_to_copy <= self.shape()[axis]);
        assert!(src_range.end <= source.shape()[axis]);
        let source_bytes = source.as_bytes();
        let destination_bytes = self.as_bytes_mut();
        let copy_bytes = rows_to_copy * block_size_bytes;

        for block_index in 0..num_blocks {
            let source_block_start = block_index * source_stride_bytes;
            let destination_block_start = block_index * destination_stride_bytes;

            let source_start = source_block_start + src_range.start * block_size_bytes;
            let destination_start = destination_block_start + dst_offset * block_size_bytes;

            let source_slice = &source_bytes[source_start..source_start + copy_bytes];
            let destination_slice = &mut destination_bytes[destination_start..destination_start + copy_bytes];
            destination_slice.copy_from_slice(source_slice);
        }
    }

    pub fn copy_from_view<T: ArrayElement, D: Dimension>(
        &mut self,
        view: ArrayView<T, D>,
    ) {
        assert_eq!(self.data_type(), T::data_type());

        let destination_slice = self.as_slice_mut::<T>();

        if let Some(source_slice) = view.as_slice_memory_order() {
            assert!(source_slice.len() <= destination_slice.len());
            destination_slice[..source_slice.len()].copy_from_slice(source_slice);
        } else {
            assert!(view.len() <= destination_slice.len());
            for (destination, source) in destination_slice.iter_mut().zip(view.iter()) {
                *destination = *source;
            }
        }
    }

    fn validate_element_type<T: ArrayElement>(&self) {
        assert_eq!(
            T::data_type(),
            self.data_type(),
            "Invalid data type, expected {:?}, actual {:?}",
            T::data_type(),
            self.data_type()
        );
    }
}

pub fn allocation_copy_from_slice<B: Backend, T: ArrayElement>(
    allocation: &mut Allocation<B>,
    data: &[T],
) -> Result<(), AllocationAccessError> {
    let bytes = bytemuck::cast_slice(data);
    if bytes.is_empty() {
        return Ok(());
    }
    let destination = allocation_as_bytes_mut(allocation);
    if bytes.len() > destination.len() {
        return Err(AllocationAccessError::WriteExceedsRange {
            write_len: bytes.len(),
            range_len: destination.len(),
        });
    }
    destination[..bytes.len()].copy_from_slice(bytes);
    Ok(())
}

pub fn allocation_as_bytes<B: Backend>(allocation: &Allocation<B>) -> &[u8] {
    let buffer_range = allocation.as_buffer_range_ref();
    let range = buffer_range.range();
    unsafe {
        std::slice::from_raw_parts(
            (buffer_range.buffer().cpu_ptr().as_ptr() as *const u8).add(range.start),
            range.len(),
        )
    }
}

pub fn allocation_as_bytes_mut<B: Backend>(allocation: &mut Allocation<B>) -> &mut [u8] {
    let buffer_range = allocation.as_buffer_range_mut();
    let range = buffer_range.range();
    unsafe {
        std::slice::from_raw_parts_mut(
            (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start),
            range.len(),
        )
    }
}

pub fn try_allocation_to_vec<B: Backend, T: ArrayElement>(
    allocation: &Allocation<B>
) -> Result<Vec<T>, AllocationAccessError> {
    let element_size = size_of::<T>();
    let allocation_bytes = allocation_as_bytes(allocation);
    if allocation_bytes.len() % element_size != 0 {
        return Err(AllocationAccessError::LengthNotMultiple {
            byte_len: allocation_bytes.len(),
            element_size,
        });
    }

    let base = allocation_bytes.as_ptr() as *const T;
    let element_count = allocation_bytes.len() / element_size;
    Ok((0..element_count).map(|index| unsafe { base.add(index).read_unaligned() }).collect())
}

pub fn allocation_to_vec<B: Backend, T: ArrayElement>(allocation: &Allocation<B>) -> Vec<T> {
    try_allocation_to_vec(allocation).expect("Failed to read allocation")
}
