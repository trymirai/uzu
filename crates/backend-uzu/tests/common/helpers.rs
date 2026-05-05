use std::rc::Rc;

use backend_uzu::backends::common::{Backend, Context};

#[allow(dead_code)]
pub fn alloc_buffer<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> B::DenseBuffer {
    context.create_buffer(elements_count * size_of::<T>()).expect("Failed to create buffer")
}

pub fn alloc_buffer_with_data<B: Backend, T: bytemuck::NoUninit>(
    context: &B::Context,
    data: &[T],
) -> B::DenseBuffer {
    context.create_buffer_with_data(bytemuck::cast_slice(data)).expect("Failed to create buffer")
}

#[allow(dead_code)]
pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().expect(format!("Failed to create context for {}", std::any::type_name::<B>()).as_str())
}
