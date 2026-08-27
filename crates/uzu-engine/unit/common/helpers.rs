use std::{mem::size_of, sync::Arc};

use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, Buffer, BufferCpuAccessible, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, SparseBuffer,
    },
};

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
macro_rules! for_each_backend {
    (|$B:ident| $body:expr) => {{
        {
            type $B = crate::backends::cpu::Cpu;
            $body
        }
        #[cfg(backend = "metal")]
        {
            type $B = crate::backends::metal::Metal;
            $body
        }
    }};
}
pub(crate) use for_each_backend;

macro_rules! for_each_non_cpu_backend {
    (|$B:ident| $body:expr) => {{
        #[cfg(backend = "metal")]
        {
            type $B = crate::backends::metal::Metal;
            $body
        }
        {
            if false {
                type $B = crate::backends::cpu::Cpu;
                $body
            }
        }
    }};
}
pub(crate) use for_each_non_cpu_backend;

pub fn buffer_size_bytes<T>(elements_count: usize) -> usize {
    elements_count * size_of::<T>()
}

pub fn create_buffer<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> B::GlobalBuffer {
    context.create_buffer(buffer_size_bytes::<T>(elements_count)).expect("Failed to create buffer")
}

pub fn create_buffer_with_data<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> B::GlobalBuffer {
    let mut buffer = context.create_buffer(buffer_size_bytes::<T>(data.len())).expect("Failed to create buffer");
    write_buffer(&mut buffer, data);
    buffer
}

pub fn buffer_to_vec<B: Backend, T: ArrayElement>(
    buffer: impl BufferRef<Backend = B, Buffer: BufferCpuAccessible>
) -> Vec<T> {
    buffer.copyout()
}

pub fn buffer_prefix_to_vec<B: Backend, T: ArrayElement>(
    buffer: impl BufferRef<Backend = B, Buffer: BufferCpuAccessible>,
    elements_count: usize,
) -> Vec<T> {
    let mut values = buffer_to_vec::<B, T>(buffer);
    values.truncate(elements_count);
    values
}

pub fn write_buffer<B: Backend, T: ArrayElement>(
    buffer: impl BufferMut<Backend = B, Buffer: BufferCpuAccessible>,
    data: &[T],
) {
    let bytes = bytemuck::cast_slice(data);
    let destination = buffer.as_slice_mut::<u8>();
    assert!(bytes.len() <= destination.len(), "source data is larger than destination buffer");
    destination[..bytes.len()].copy_from_slice(bytes);
}

pub fn create_context<B: Backend>() -> Arc<<B as Backend>::Context> {
    B::Context::new().unwrap_or_else(|_| panic!("Failed to create context for {}", std::any::type_name::<B>()))
}

pub fn submit_command_buffer<E: CommandBufferEncoding>(command_buffer: E) {
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
}

pub fn sparse_buffer_create<B: Backend>(
    context: &B::Context,
    capacity: usize,
) -> B::SparseBuffer {
    context.create_sparse_buffer(capacity).expect("Failed to create sparse buffer")
}

pub fn sparse_buffer_create_and_map<B: Backend>(
    context: &B::Context,
    capacity: usize,
) -> B::SparseBuffer {
    let mut buffer = sparse_buffer_create::<B>(context, capacity);
    buffer.map(context, 0..buffer.total_pages()).expect("Failed to map sparse buffer");
    buffer
}

pub fn sparse_buffer_create_with<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> B::SparseBuffer {
    let capacity_bytes = buffer_size_bytes::<T>(data.len());
    let mut buffer = sparse_buffer_create_and_map::<B>(context, capacity_bytes);
    sparse_buffer_write::<B, T>(context, &mut buffer, data);
    buffer
}

pub fn buffer_readback<B: Backend>(
    context: &B::Context,
    buffer: impl BufferRef<Backend = B>,
) -> B::GlobalBuffer {
    let mut output_buffer = create_buffer::<B, u8>(context, buffer.size());

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    command_buffer.encode_copy(buffer, &mut output_buffer);
    submit_command_buffer(command_buffer);

    output_buffer
}

pub fn sparse_buffer_read_vec<B: Backend, T: ArrayElement>(
    context: &B::Context,
    buffer: impl BufferRef<Backend = B>,
    elements_count: usize,
) -> Vec<T> {
    let dense_buffer =
        buffer_readback::<B>(context, buffer.subrange(..elements_count * T::data_type().size_in_bytes()));
    buffer_to_vec(&dense_buffer)
}

pub fn sparse_buffer_write<B: Backend, T: ArrayElement>(
    context: &B::Context,
    buffer: impl BufferMut<Backend = B>,
    data: &[T],
) {
    let data_buffer = create_buffer_with_data::<B, T>(context, data);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    command_buffer.encode_copy(&data_buffer, buffer.subrange_mut(..data_buffer.size()));
    submit_command_buffer(command_buffer);
}
