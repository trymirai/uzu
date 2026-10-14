use std::any::Any;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, CommandBuffer, Context, DeviceCapabilities, Kernels, SparseBuffer,
        gpu_types::kv_cache_update::Copy,
        kernel::{KVCacheUpdateInt8Kernel, KVCacheUpdateKernel, KVFormat, kv_layout::Int8KVLayout},
    },
    data_type::DataType,
    utils::size_for_shape::size_for_shape,
};

pub enum KVStorage<B: Backend> {
    BF16(BF16KV<B>, <B::Kernels as Kernels>::KVCacheUpdateKernel),
    Int8(Int8KV<B>, <B::Kernels as Kernels>::KVCacheUpdateInt8Kernel),
}

pub struct BF16KV<B: Backend> {
    pub keys: Box<dyn Buffer<Backend = B>>,
    pub values: Box<dyn Buffer<Backend = B>>,
    pub element_dim: u32,
    pub data_type: DataType,
}

pub struct Int8KV<B: Backend> {
    pub layout: Int8KVLayout,
    pub k_codes: Box<dyn Buffer<Backend = B>>,
    pub v_codes: Box<dyn Buffer<Backend = B>>,
    pub k_scales: Box<dyn Buffer<Backend = B>>,
    pub v_scales: Box<dyn Buffer<Backend = B>>,
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
                let layout = Int8KVLayout::new(head_dim, num_kv_heads);
                let scales = || -> Result<Box<dyn Buffer<Backend = B>>, B::Error> {
                    let bytes = layout.scale_tile_bytes(max_elements);
                    if context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
                        return create_kv_buffer(context, bytes);
                    }
                    let mut buffer = context.create_buffer(bytes)?;
                    buffer.as_slice_mut::<u8>().fill(0);
                    Ok(Box::new(buffer))
                };
                let kv = Int8KV {
                    k_codes: create_kv_buffer(context, layout.tile_bytes(max_elements))?,
                    v_codes: create_kv_buffer(context, layout.tile_bytes(max_elements))?,
                    k_scales: scales()?,
                    v_scales: scales()?,
                    layout,
                };
                Self::Int8(kv, <B::Kernels as Kernels>::KVCacheUpdateInt8Kernel::new(context, head_dim, num_kv_heads)?)
            },
            KVFormat::BF16 => {
                let element_dim = num_kv_heads * head_dim;
                let bytes = size_for_shape(&[max_elements, element_dim], data_type);
                let kv = BF16KV {
                    keys: create_kv_buffer(context, bytes)?,
                    values: create_kv_buffer(context, bytes)?,
                    element_dim,
                    data_type,
                };
                Self::BF16(kv, <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, data_type)?)
            },
        })
    }

    pub fn prepare(
        &mut self,
        elements_required: u32,
    ) -> Result<(), B::Error> {
        let map = |buffer: &mut Box<dyn Buffer<Backend = B>>, bytes: usize| -> Result<(), B::Error> {
            match (buffer.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>() {
                Some(sparse) => sparse.map(bytes),
                None => Ok(()),
            }
        };
        match self {
            Self::BF16(kv, _) => {
                let bytes = size_for_shape(&[elements_required, kv.element_dim], kv.data_type);
                map(&mut kv.keys, bytes)?;
                map(&mut kv.values, bytes)
            },
            Self::Int8(kv, _) => {
                let code_bytes = kv.layout.tile_bytes(elements_required);
                map(&mut kv.k_codes, code_bytes)?;
                map(&mut kv.v_codes, code_bytes)?;
                let scale_bytes = kv.layout.scale_tile_bytes(elements_required);
                map(&mut kv.k_scales, scale_bytes)?;
                map(&mut kv.v_scales, scale_bytes)
            },
        }
    }

    pub fn encode_accept(
        &mut self,
        copies: &[Copy],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let count = copies.len() as u32;
        match self {
            Self::BF16(kv, update) => {
                update.encode(kv.keys.as_mut(), kv.values.as_mut(), copies, count, kv.element_dim, command_buffer)
            },
            Self::Int8(kv, update) => update.encode(
                kv.k_codes.as_mut(),
                kv.v_codes.as_mut(),
                kv.k_scales.as_mut(),
                kv.v_scales.as_mut(),
                copies,
                count,
                command_buffer,
            ),
        }
    }
}
