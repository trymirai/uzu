use std::rc::Rc;

use backend_uzu::backends::common::{Backend, Context, DenseBuffer};

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
    if data.len() == 0 {
        // Metal doesn't allow creating 0-byte buffers, create a minimal buffer instead
        return context.create_buffer(1).expect("Failed to create buffer");
    }

    let slice: &[u8] = bytemuck::cast_slice(data);
    let buffer = context.create_buffer(slice.len()).expect("Failed to create buffer");
    let bytes = unsafe { std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr() as *mut u8, buffer.length()) };
    bytes.copy_from_slice(slice);
    buffer
}

#[allow(dead_code)]
pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().expect(format!("Failed to create context for {}", std::any::type_name::<B>()).as_str())
}
