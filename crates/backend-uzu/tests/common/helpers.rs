use std::rc::Rc;

use backend_uzu::{
    ArrayElement, allocation_copy_from_slice, allocation_from_slice,
    backends::common::{Allocation, AllocationType, Backend, Context},
};

pub fn alloc_allocation<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> Allocation<B> {
    let byte_len = (elements_count * size_of::<T>()).max(1);
    context.create_allocation(byte_len, AllocationType::Global).expect("Failed to create allocation")
}

pub fn alloc_allocation_with_data<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    allocation_from_slice(context, data)
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
    allocation: &Allocation<B>,
    data: &[T],
) {
    allocation_copy_from_slice(allocation, data)
}

pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().expect(format!("Failed to create context for {}", std::any::type_name::<B>()).as_str())
}
