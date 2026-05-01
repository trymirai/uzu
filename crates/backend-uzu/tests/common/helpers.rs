use std::{mem::size_of, rc::Rc};

use backend_uzu::{
    ArrayElement, allocation_copy_from_slice,
    backends::common::{Allocation, AllocationType, Backend, Context, Encoder},
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
