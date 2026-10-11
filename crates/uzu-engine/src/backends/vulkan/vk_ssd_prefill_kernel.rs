use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;

use super::{Error, SSDPrefillVulkanKernel, VkBuffer, VkCommandBufferEncoding, VkContext, Vulkan};
use crate::{
    backends::common::{BufferMut, BufferRef, kernel::SSDPrefillKernel},
    data_type::DataType,
};

pub struct VkSSDPrefillKernel {
    context: Arc<VkContext>,
    data_type: DataType,
    kernels: Mutex<HashMap<u32, SSDPrefillVulkanKernel>>,
}

impl SSDPrefillKernel for VkSSDPrefillKernel {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        data_type: DataType,
    ) -> Result<Self, Error> {
        Ok(Self {
            context: context.arc(),
            data_type,
            kernels: Mutex::new(HashMap::new()),
        })
    }

    fn encode(
        &self,
        x: impl BufferRef<Backend = Vulkan>,
        dt_raw: impl BufferRef<Backend = Vulkan>,
        b: impl BufferRef<Backend = Vulkan>,
        c: impl BufferRef<Backend = Vulkan>,
        d: impl BufferRef<Backend = Vulkan>,
        z: impl BufferRef<Backend = Vulkan>,
        state: impl BufferMut<Backend = Vulkan>,
        y: impl BufferMut<Backend = Vulkan>,
        suffix_len: u32,
        group_size: u32,
        state_size: u32,
        x_strides: &[u32; 3],
        dt_strides: &[u32; 2],
        cb_strides: &[u32; 3],
        state_strides: &[u32; 3],
        num_heads: u32,
        head_dim: u32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        if suffix_len == 0 || num_heads == 0 || head_dim == 0 {
            return;
        }
        let mut kernels = self.kernels.lock();
        let kernel = kernels.entry(state_size).or_insert_with(|| {
            SSDPrefillVulkanKernel::new(&self.context, self.data_type, state_size)
                .expect("SSDPrefill: cannot create the pipeline for this data type and state_size")
        });
        let (x, x_range) = BufferRef::parts(x);
        let (dt_raw, dt_raw_range) = BufferRef::parts(dt_raw);
        let (b, b_range) = BufferRef::parts(b);
        let (c, c_range) = BufferRef::parts(c);
        let (d, d_range) = BufferRef::parts(d);
        let (z, z_range) = BufferRef::parts(z);
        let (state, state_range) = BufferMut::parts(state);
        let (y, y_range) = BufferMut::parts(y);
        // SAFETY: as for the CPU kernels, the common contract makes callers pass ranges that cover every element the
        // kernel indexes, y and the state aliasing no other argument.
        unsafe {
            SSDPrefillVulkanKernel::encode(
                kernel,
                VkBuffer::resolve(x, x_range),
                VkBuffer::resolve(dt_raw, dt_raw_range),
                VkBuffer::resolve(b, b_range),
                VkBuffer::resolve(c, c_range),
                VkBuffer::resolve(d, d_range),
                VkBuffer::resolve(z, z_range),
                VkBuffer::resolve(state, state_range),
                VkBuffer::resolve(y, y_range),
                suffix_len,
                group_size,
                x_strides,
                dt_strides,
                cb_strides,
                state_strides,
                num_heads,
                head_dim,
                command_buffer,
            )
        }
    }
}
