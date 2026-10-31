use std::{mem::size_of, sync::Arc};

use futures::executor::block_on;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
        GlobalBufferCpuGpu, kernel::AttentionKernelConfig,
    },
    data_type::DataTypeRepr,
};
#[cfg(backend = "vulkan")]
use crate::{backends::vulkan, utils::downcast::downcast_ref};

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
macro_rules! for_each_backend {
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
            type $B = crate::backends::cpu::Cpu;
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
) -> B::GlobalGpuBuffer {
    context.create_buffer_gpu(buffer_size_bytes::<T>(elements_count)).expect("Failed to create buffer")
}

pub fn create_buffer_with_data<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    data: &[T],
) -> B::GlobalGpuBuffer {
    let mut buffer =
        context.create_buffer_cpu_gpu(buffer_size_bytes::<T>(data.len())).expect("Failed to create buffer");
    buffer.copyin(data);
    buffer.handoff()
}

pub fn buffer_to_vec<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    buffer: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
) -> Vec<T> {
    block_on(context.copyout(buffer)).expect("Failed to read buffer")
}

pub fn buffer_prefix_to_vec<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    buffer: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
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

/// `kernel` once built, or `None` where its backend does not support it, so a shared test skips that backend. Vulkan
/// builds the kernels it lacks as `Unsupported`, which fails with `NotSupported`.
pub fn supported_kernel<B: Backend, K>(kernel: Result<K, B::Error>) -> Option<K> {
    #[cfg(backend = "vulkan")]
    if let Err(error) = &kernel
        && let Some(vulkan::Error::NotSupported) = downcast_ref::<vulkan::Error>(error)
    {
        return None;
    }
    Some(kernel.expect("kernel"))
}

pub fn submit_command_buffer<E: CommandBufferEncoding>(command_buffer: E) {
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
}
