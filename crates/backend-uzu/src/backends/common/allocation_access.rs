use thiserror::Error;

use crate::{
    ArrayElement,
    backends::common::{Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, DenseBuffer},
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AllocationAccessError {
    #[error("allocation write length {write_len} exceeds allocation length {range_len}")]
    WriteExceedsRange {
        write_len: usize,
        range_len: usize,
    },
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

pub fn allocation_to_vec<B: Backend, T: ArrayElement>(allocation: &Allocation<B>) -> Vec<T> {
    bytemuck::cast_slice(allocation_as_bytes(allocation)).to_vec()
}
