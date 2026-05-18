use std::{mem::size_of, rc::Rc};

use backend_uzu::{
    ArrayElement, allocation_copy_from_slice,
    backends::{
        common::{Allocation, AllocationType, Backend, Buffer, Context, Encoder, SparseBuffer},
        metal::Metal,
    },
    prelude::MetalContext,
};

pub fn allocation_size_bytes<T>(elements_count: usize) -> usize {
    elements_count * size_of::<T>()
}

pub fn alloc_allocation<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> Allocation<B> {
    context
        .create_allocation(allocation_size_bytes::<T>(elements_count), AllocationType::Global)
        .expect("Failed to create allocation")
}

pub fn alloc_allocation_with_data<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    let mut allocation = context
        .create_allocation(allocation_size_bytes::<T>(data.len()), AllocationType::Global)
        .expect("Failed to create allocation");
    allocation_copy_from_slice(&mut allocation, data).expect("Failed to initialize allocation");
    allocation
}

pub fn allocation_to_vec<B: Backend, T: ArrayElement>(allocation: &Allocation<B>) -> Vec<T> {
    backend_uzu::allocation_to_vec(allocation)
}

pub fn allocation_prefix_to_vec<B: Backend, T: ArrayElement>(
    allocation: &Allocation<B>,
    elements_count: usize,
) -> Vec<T> {
    let mut values = allocation_to_vec::<B, T>(allocation);
    values.truncate(elements_count);
    values
}

pub fn write_allocation<B: Backend, T: ArrayElement>(
    allocation: &mut Allocation<B>,
    data: &[T],
) {
    allocation_copy_from_slice(allocation, data).expect("Failed to write allocation")
}

pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().expect(format!("Failed to create context for {}", std::any::type_name::<B>()).as_str())
}

pub fn submit_encoder<B: Backend>(encoder: Encoder<B>) {
    encoder.end_encoding().submit().wait_until_completed().unwrap();
}

pub fn sparse_buffer_create<B: Backend>(
    context: &B::Context,
    capacity: usize,
) -> B::SparseBuffer {
    context.create_sparse_buffer(capacity).expect("Failed to create sparse buffer")
}

pub fn sparse_buffer_create_mapped<B: Backend>(
    context: &B::Context,
    capacity: usize,
) -> B::SparseBuffer {
    let mut buffer = sparse_buffer_create::<B>(context, capacity);
    let total_pages = buffer.size() / buffer.page_size_bytes();
    buffer.map(context, &(0..total_pages)).expect("Failed to map sparse buffer");
    context.sparse_mappings_signal();
    buffer
}

pub fn sparse_buffer_create_with<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> B::SparseBuffer {
    let capacity_bytes = allocation_size_bytes::<T>(data.len());
    let mut buffer = sparse_buffer_create_mapped::<B>(context, capacity_bytes);
    sparse_buffer_write::<B, T>(context, &mut buffer, data);
    buffer
}

pub fn sparse_buffer_read_allocation<B: Backend>(
    context: &B::Context,
    buffer: &B::SparseBuffer,
    size: usize,
) -> Allocation<B> {
    let mut allocation = alloc_allocation::<B, u8>(context, size);
    let range = 0..size;

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");
    encoder.encode_copy(buffer, range.clone(), &mut allocation, range.clone());
    submit_encoder(encoder);

    allocation
}

pub fn sparse_buffer_read_vec<B: Backend, T: ArrayElement>(
    context: &B::Context,
    buffer: &B::SparseBuffer,
    elements_count: usize,
) -> Vec<T> {
    let allocation =
        sparse_buffer_read_allocation::<B>(context, buffer, elements_count * T::data_type().size_in_bytes());
    allocation_to_vec(&allocation)
}

pub fn sparse_buffer_write<B: Backend, T: ArrayElement>(
    context: &B::Context,
    buffer: &mut B::SparseBuffer,
    data: &[T],
) {
    let data_allocation = alloc_allocation_with_data::<B, T>(context, data);
    let data_range = 0..allocation_size_bytes::<T>(data.len());

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");
    encoder.encode_copy(&data_allocation, data_range.clone(), buffer, data_range.clone());
    submit_encoder(encoder);
}
