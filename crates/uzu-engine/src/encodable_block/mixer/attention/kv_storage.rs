use std::any::Any;

use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, Buffer, BufferMut, CommandBuffer, Context, DeviceCapabilities, Kernels, SparseBuffer,
        gpu_types::Copy,
        kernel::{KVCacheUpdateInt8Kernel, KVCacheUpdateKernel, KVFormat, kv_layout::Int8KVLayout},
    },
    data_type::DataType,
};

pub enum KVStorage<B: Backend> {
    /// K/V rows in the model's data type.
    BF16 {
        keys: Box<dyn Buffer<Backend = B>>,
        values: Box<dyn Buffer<Backend = B>>,
        update: <B::Kernels as Kernels>::KVCacheUpdateKernel,
        element_dim: u32,
        data_type: DataType,
    },
    Int8(Int8KV<B>, <B::Kernels as Kernels>::KVCacheUpdateInt8Kernel),
}

/// K/V codes are sparse where supported; scales are dense and zeroed.
pub struct Int8KV<B: Backend> {
    pub layout: Int8KVLayout,
    pub k_codes: Box<dyn Buffer<Backend = B>>,
    pub v_codes: Box<dyn Buffer<Backend = B>>,
    pub k_scales: B::GlobalBuffer,
    pub v_scales: B::GlobalBuffer,
}

fn create_kv_buffer<B: Backend>(
    context: &B::Context,
    bytes: usize,
) -> Result<Box<dyn Buffer<Backend = B>>, B::Error> {
    Ok(if context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
        Box::new(context.create_sparse_buffer(bytes)?)
    } else {
        Box::new(context.create_buffer(bytes)?)
    })
}

impl<B: Backend> KVStorage<B> {
    pub fn create_empty(
        format: KVFormat,
        num_kv_heads: u32,
        head_dim: u32,
        max_elements: u32,
        data_type: DataType,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        Ok(match format {
            KVFormat::Int8 => {
                let layout = Int8KVLayout::new(head_dim, num_kv_heads, max_elements);
                // The kernel reads a scale <= 0 as never written.
                let zeroed = || -> Result<B::GlobalBuffer, B::Error> {
                    let mut buffer = context.create_buffer(2 * layout.scale_count())?;
                    buffer.as_slice_mut::<u8>().fill(0);
                    Ok(buffer)
                };
                let kv = Int8KV {
                    k_codes: create_kv_buffer(context, layout.code_bytes())?,
                    v_codes: create_kv_buffer(context, layout.code_bytes())?,
                    k_scales: zeroed()?,
                    v_scales: zeroed()?,
                    layout,
                };
                Self::Int8(kv, <B::Kernels as Kernels>::KVCacheUpdateInt8Kernel::new(context, head_dim, num_kv_heads)?)
            },
            KVFormat::BF16 => {
                let element_dim = num_kv_heads * head_dim;
                let bytes = size_for_shape(&[max_elements, element_dim], data_type);
                Self::BF16 {
                    keys: create_kv_buffer(context, bytes)?,
                    values: create_kv_buffer(context, bytes)?,
                    update: <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, data_type)?,
                    element_dim,
                    data_type,
                }
            },
        })
    }

    pub fn prepare(
        &mut self,
        elements_required: u32,
    ) -> Result<(), B::Error> {
        let (buffers, bytes_required) = match self {
            Self::BF16 {
                keys,
                values,
                element_dim,
                data_type,
                ..
            } => ([keys, values], size_for_shape(&[elements_required, *element_dim], *data_type)),
            Self::Int8(kv, _) => ([&mut kv.k_codes, &mut kv.v_codes], kv.layout.tile_bytes(elements_required)),
        };
        for buffer in buffers {
            if let Some(sparse) = (buffer.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>() {
                sparse.map(bytes_required)?;
            }
        }
        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        copies: &[Copy],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let count = copies.len() as u32;
        match self {
            Self::BF16 {
                keys,
                values,
                update,
                element_dim,
                ..
            } => update.encode(keys.as_mut(), values.as_mut(), copies, count, *element_dim, command_buffer),
            Self::Int8(kv, update) => update.encode(
                kv.k_codes.as_mut(),
                kv.v_codes.as_mut(),
                &mut kv.k_scales,
                &mut kv.v_scales,
                copies,
                count,
                kv.layout.padded_capacity,
                command_buffer,
            ),
        }
    }
}
