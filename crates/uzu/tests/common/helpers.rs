use std::rc::Rc;

use uzu::{
    ArrayElement, allocation_as_slice,
    backends::common::{Allocation, AllocationType, Backend, Buffer, Context},
};

pub fn alloc_allocation<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> Allocation<B> {
    let byte_len = (elements_count * size_of::<T>()).max(1);
    context.create_allocation(byte_len, AllocationType::Global).expect("Failed to create allocation")
}

pub fn alloc_allocation_with_data<B: Backend, T: bytemuck::NoUninit>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    let allocation = alloc_allocation::<B, T>(context, data.len());
    if data.is_empty() {
        return allocation;
    }

    let bytes: &[u8] = bytemuck::cast_slice(data);
    let (buffer, range) = allocation.as_buffer_range();
    let destination =
        unsafe { std::slice::from_raw_parts_mut((buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start), bytes.len()) };
    destination.copy_from_slice(bytes);
    allocation
}

pub fn allocation_to_vec<B: Backend, T: ArrayElement>(allocation: &Allocation<B>) -> Vec<T> {
    allocation_as_slice::<T, B>(allocation).to_vec()
}

pub fn allocation_prefix_to_vec<B: Backend, T: ArrayElement>(
    allocation: &Allocation<B>,
    elements_count: usize,
) -> Vec<T> {
    let mut values = allocation_to_vec::<B, T>(allocation);
    values.truncate(elements_count);
    values
}

pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().expect(format!("Failed to create context for {}", std::any::type_name::<B>()).as_str())
}
