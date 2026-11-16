use std::{mem::size_of, path::Path, sync::Arc};

use futures::executor::block_on;
use serde_json::{Map, Value, json};

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
        GlobalBufferCpuGpu,
    },
    data_type::DataTypeRepr,
};

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
macro_rules! for_each_backend {
    (|$B:ident| $body:expr) => {{
        #[cfg(backend = "metal")]
        {
            type $B = crate::backends::metal::Metal;
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
    buffer.handoff().expect("Failed to hand off buffer")
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

pub fn submit_command_buffer<E: CommandBufferEncoding>(command_buffer: E) {
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
}

pub fn buffer_readback<B: Backend>(
    context: &B::Context,
    buffer: impl BufferRef<Backend = B>,
) -> B::GlobalGpuBuffer {
    let mut output_buffer = create_buffer::<B, u8>(context, buffer.size());

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    command_buffer.encode_copy(buffer, &mut output_buffer);
    submit_command_buffer(command_buffer);

    output_buffer
}

#[derive(Default)]
pub struct SafetensorsWriter {
    metadata: Map<String, Value>,
    tensors: Map<String, Value>,
    data: Vec<u8>,
}

impl SafetensorsWriter {
    pub fn add_metadata(
        &mut self,
        key: String,
        value: &Value,
    ) {
        self.metadata.insert(key, value.to_string().into());
    }

    pub fn add_tensor(
        &mut self,
        name: String,
        dtype: &str,
        shape: &[u32],
        bytes: &[u8],
    ) {
        let data_offsets = [self.data.len(), self.data.len() + bytes.len()];
        self.tensors.insert(name, json!({"dtype": dtype, "shape": shape, "data_offsets": data_offsets}));
        self.data.extend_from_slice(bytes);
    }

    pub fn write(
        mut self,
        path: &Path,
    ) {
        self.tensors.insert("__metadata__".into(), self.metadata.into());
        let header = serde_json::to_vec(&self.tensors).expect("safetensors header");
        std::fs::write(path, [&(header.len() as u64).to_le_bytes()[..], &header, &self.data].concat())
            .expect("write safetensors");
    }
}
