use std::{mem::size_of, sync::Arc};

use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, BufferCpuAccessible, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context,
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
    buffer.copyin(data);
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

pub fn create_context<B: Backend>() -> Arc<<B as Backend>::Context> {
    B::Context::new().unwrap_or_else(|_| panic!("Failed to create context for {}", std::any::type_name::<B>()))
}

pub fn submit_command_buffer<E: CommandBufferEncoding>(command_buffer: E) {
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
}

pub fn buffer_readback<B: Backend>(
    context: &B::Context,
    buffer: impl BufferRef<Backend = B>,
) -> B::GlobalBuffer {
    let mut output_buffer = create_buffer::<B, u8>(context, buffer.size());

    let mut command_buffer = context.create_command_buffer(None, None, false).expect("Failed to create command buffer");
    command_buffer.encode_copy(buffer, &mut output_buffer);
    submit_command_buffer(command_buffer);

    output_buffer
}
