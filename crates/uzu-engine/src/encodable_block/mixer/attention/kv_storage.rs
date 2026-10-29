use std::any::Any;

use crate::{
    backends::common::{
        Backend, CommandBuffer, Context, DeviceCapabilities, GpuBuffer, Kernels, SparseBuffer,
        kernel::{KVCacheCompactInt8Kernel, KVCacheCompactKernel, KVFormat, kv_layout::Int8KVLayout},
    },
    data_type::DataType,
    utils::size_for_shape::size_for_shape,
};

pub enum KVStorage<B: Backend> {
    BF16(BF16KV<B>, <B::Kernels as Kernels>::KVCacheCompactKernel),
    Int8(Int8KV<B>, <B::Kernels as Kernels>::KVCacheCompactInt8Kernel),
}

pub struct BF16KV<B: Backend> {
    pub keys: Box<dyn GpuBuffer<Backend = B>>,
    pub values: Box<dyn GpuBuffer<Backend = B>>,
    pub element_dim: u32,
    pub data_type: DataType,
}

pub struct Int8KV<B: Backend> {
    pub layout: Int8KVLayout,
    pub k_codes: Box<dyn GpuBuffer<Backend = B>>,
    pub v_codes: Box<dyn GpuBuffer<Backend = B>>,
    pub k_scales: Box<dyn GpuBuffer<Backend = B>>,
    pub v_scales: Box<dyn GpuBuffer<Backend = B>>,
}

fn create_kv_buffer<B: Backend>(
    context: &B::Context,
    bytes: usize,
) -> Result<Box<dyn GpuBuffer<Backend = B>>, B::Error> {
    Ok(if context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
        Box::new(context.create_sparse_buffer(bytes)?)
    } else {
        Box::new(context.create_buffer_gpu(bytes)?)
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
            KVFormat::INT8 => {
                let layout = Int8KVLayout::new(head_dim, num_kv_heads);
                let code_bytes = layout.code_bytes(max_elements);
                let scale_bytes = layout.scale_bytes(max_elements);
                let kv = Int8KV {
                    k_codes: create_kv_buffer(context, code_bytes)?,
                    v_codes: create_kv_buffer(context, code_bytes)?,
                    k_scales: create_kv_buffer(context, scale_bytes)?,
                    v_scales: create_kv_buffer(context, scale_bytes)?,
                    layout,
                };
                Self::Int8(kv, <B::Kernels as Kernels>::KVCacheCompactInt8Kernel::new(context, head_dim, num_kv_heads)?)
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
                Self::BF16(kv, <B::Kernels as Kernels>::KVCacheCompactKernel::new(context, data_type)?)
            },
        })
    }

    pub fn prepare(
        &mut self,
        elements_required: u32,
    ) -> Result<(), B::Error> {
        let map = |buffer: &mut dyn GpuBuffer<Backend = B>, bytes: usize| -> Result<(), B::Error> {
            match (buffer as &mut dyn Any).downcast_mut::<B::SparseBuffer>() {
                Some(sparse) => sparse.map(bytes),
                None => Ok(()),
            }
        };
        match self {
            Self::BF16(kv, _) => {
                let bytes = size_for_shape(&[elements_required, kv.element_dim], kv.data_type);
                map(kv.keys.as_mut(), bytes)?;
                map(kv.values.as_mut(), bytes)
            },
            Self::Int8(kv, _) => {
                let code_bytes = kv.layout.code_bytes(elements_required);
                map(kv.k_codes.as_mut(), code_bytes)?;
                map(kv.v_codes.as_mut(), code_bytes)?;
                let scale_bytes = kv.layout.scale_bytes(elements_required);
                map(kv.k_scales.as_mut(), scale_bytes)?;
                map(kv.v_scales.as_mut(), scale_bytes)
            },
        }
    }

    pub fn encode_compact(
        &mut self,
        accepted_indices: &B::GlobalGpuBuffer,
        suffix_base: u32,
        ring_write_start: u32,
        ring_capacity: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        match self {
            Self::BF16(kv, compact) => compact.encode(
                kv.keys.as_mut(),
                kv.values.as_mut(),
                accepted_indices,
                suffix_base,
                ring_write_start,
                ring_capacity,
                kv.element_dim,
                command_buffer,
            ),
            Self::Int8(kv, compact) => compact.encode(
                kv.k_codes.as_mut(),
                kv.v_codes.as_mut(),
                kv.k_scales.as_mut(),
                kv.v_scales.as_mut(),
                accepted_indices,
                suffix_base,
                ring_write_start,
                ring_capacity,
                command_buffer,
            ),
        }
    }
}
