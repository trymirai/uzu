use std::ptr;

use ndarray::{ArrayView, Dimension};

use crate::{
    ArrayElement, DataType,
    array::{Array, size_for_shape},
    backends::common::{Allocation, AllocationType, Backend, Buffer, Context, Encoder},
};

pub fn create_allocation<B: Backend>(
    context: &B::Context,
    shape: &[usize],
    data_type: DataType,
) -> Allocation<B> {
    let size = size_for_shape(shape, data_type);
    context.create_allocation(size, AllocationType::Global).expect("Failed to create allocation")
}

pub fn create_zeroed_allocation<B: Backend>(
    context: &B::Context,
    shape: &[usize],
    data_type: DataType,
) -> Allocation<B> {
    let mut allocation = create_allocation(context, shape, data_type);
    fill_allocation(&mut allocation, 0);
    allocation
}

pub fn fill_allocation<B: Backend>(
    allocation: &mut Allocation<B>,
    value: u8,
) {
    unsafe {
        let dst = allocation_bytes_mut(allocation);
        dst.fill(value);
    }
}

pub fn copy_slice_to_allocation<T: ArrayElement, B: Backend>(
    allocation: &mut Allocation<B>,
    data: &[T],
) {
    let bytes = bytemuck::cast_slice(data);
    unsafe {
        let dst = allocation_bytes_mut(allocation);
        assert_eq!(dst.len(), bytes.len());
        dst.copy_from_slice(bytes);
    }
}

pub fn copy_array_to_allocation<C: Backend, B: Backend>(
    allocation: &mut Allocation<B>,
    array: &Array<C>,
) {
    unsafe {
        let dst = allocation_bytes_mut(allocation);
        assert_eq!(dst.len(), array.size());
        dst.copy_from_slice(array.as_bytes());
    }
}

pub fn copy_buffer_bytes_to_allocation<B: Backend>(
    allocation: &mut Allocation<B>,
    buffer: &B::Buffer,
    source_offset: usize,
    byte_len: usize,
) {
    unsafe {
        let dst = allocation_bytes_mut(allocation);
        assert_eq!(dst.len(), byte_len);
        let src = std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(source_offset), byte_len);
        dst.copy_from_slice(src);
    }
}

pub fn encode_copy_buffer_bytes_to_allocation<B: Backend>(
    encoder: &mut Encoder<B>,
    allocation: &mut Allocation<B>,
    buffer: &B::Buffer,
    source_offset: usize,
    byte_len: usize,
) {
    let (destination_buffer, destination_range) = allocation.as_buffer_range();
    assert_eq!(destination_range.len(), byte_len);
    encoder.encode_copy(buffer, source_offset..source_offset + byte_len, destination_buffer, destination_range);
}

#[cfg(feature = "tracing")]
pub fn encode_copy_allocation_to_array<B: Backend>(
    encoder: &mut Encoder<B>,
    source: &Allocation<B>,
    destination: &Array<B>,
) {
    let (source_buffer, source_range) = source.as_buffer_range();
    let destination_buffer = destination.buffer();
    debug_assert_eq!(destination.size(), source_range.len());

    encoder.encode_copy(source_buffer, source_range, &mut destination_buffer.borrow_mut(), 0..destination.size());
}

pub fn copy_view_to_allocation<T: ArrayElement, D: Dimension, B: Backend>(
    allocation: &mut Allocation<B>,
    view: ArrayView<T, D>,
) {
    unsafe {
        let dst = allocation_bytes_mut(allocation);
        let dst_slice: &mut [T] = bytemuck::cast_slice_mut(dst);

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

pub fn copy_allocation_to_slice<T: ArrayElement, B: Backend>(allocation: &Allocation<B>) -> &[T] {
    let (buffer, range) = allocation.as_buffer_range();
    let bytes =
        unsafe { std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start), range.len()) };
    bytemuck::cast_slice(bytes)
}

pub fn copy_allocation_to_allocation<B: Backend>(
    destination: &mut Allocation<B>,
    source: &Allocation<B>,
) {
    let (source_buffer, source_range) = source.as_buffer_range();
    unsafe {
        let dst = allocation_bytes_mut(destination);
        let src = std::slice::from_raw_parts(
            (source_buffer.cpu_ptr().as_ptr() as *const u8).add(source_range.start),
            source_range.len(),
        );
        assert_eq!(dst.len(), src.len());
        dst.copy_from_slice(src);
    }
}

unsafe fn allocation_bytes_mut<B: Backend>(allocation: &mut Allocation<B>) -> &mut [u8] {
    let (buffer, range) = allocation.as_buffer_range();
    let dst = unsafe { (buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start) };
    unsafe { ptr::slice_from_raw_parts_mut(dst, range.len()).as_mut().unwrap() }
}
