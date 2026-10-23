use std::{mem::size_of, ops::Deref, sync::Arc};

use futures::executor::block_on;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        GlobalBufferInitializer, kernel::AttentionKernelConfig,
    },
    data_type::DataTypeRepr,
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
        #[cfg(backend = "vulkan")]
        {
            type $B = crate::backends::vulkan::Vulkan;
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
        #[cfg(backend = "vulkan")]
        {
            type $B = crate::backends::vulkan::Vulkan;
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

pub fn default_attention_config<T: DataTypeRepr>(
    head_dim: usize,
    num_q_heads: usize,
    num_groups: usize,
    is_causal: bool,
) -> AttentionKernelConfig {
    AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: num_groups as u32,
        num_q_heads: num_q_heads as u32,
        has_sinks: false,
        gate_row_stride: None,
        is_kv_cache_ring: false,
        is_causal,
        sliding_window_size: None,
        scale: Some(1.0 / (head_dim as f32).sqrt()),
        data_type: T::data_type(),
    }
}

pub fn buffer_size_bytes<T>(elements_count: usize) -> usize {
    elements_count * size_of::<T>()
}

pub fn create_buffer<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> B::GlobalBuffer {
    context.create_buffer(buffer_size_bytes::<T>(elements_count)).expect("Failed to create buffer")
}

pub fn create_buffer_with_data<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    data: &[T],
) -> B::GlobalBuffer {
    let mut initializer =
        context.create_buffer_initializer(buffer_size_bytes::<T>(data.len())).expect("Failed to create buffer");
    initializer.copyin(data);
    initializer.end_initialization().expect("Failed to initialize buffer")
}

pub fn buffer_to_vec<B: Backend, T: DataTypeRepr>(
    context: &impl Deref<Target = B::Context>,
    buffer: impl BufferRef<Backend = B, Buffer = B::GlobalBuffer>,
) -> Vec<T> {
    block_on(context.deref().read_completed(buffer)).expect("Failed to read buffer")
}

pub fn buffer_prefix_to_vec<B: Backend, T: DataTypeRepr>(
    context: &impl Deref<Target = B::Context>,
    buffer: impl BufferRef<Backend = B, Buffer = B::GlobalBuffer>,
    elements_count: usize,
) -> Vec<T> {
    let mut values = buffer_to_vec::<B, T>(context, buffer);
    values.truncate(elements_count);
    values
}

pub fn create_context<B: Backend>() -> Arc<<B as Backend>::Context> {
    block_on(B::Context::new())
        .unwrap_or_else(|_| panic!("Failed to create context for {}", std::any::type_name::<B>()))
}

pub fn submit_command_buffer<E: CommandBufferEncoding>(command_buffer: E) {
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
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
