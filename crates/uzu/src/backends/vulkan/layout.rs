use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::{buffer::VkBuffer, context::VkContext};

pub struct VkComputeShaderLayoutSet {
    device: Arc<ash::Device>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
}

impl VkComputeShaderLayoutSet {
    pub fn new(
        ctx: &VkContext,
        layout_buffers: &[VkComputeShaderLayoutBuffer],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let layout_bindings = layout_buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(buffer.binding)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();

        // descriptor set
        let descriptor_set_layout = {
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
            unsafe { ctx.device().create_descriptor_set_layout(&info, None)? }
        };
        let descriptor_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(layout_buffers.len() as u32)];
        let descriptor_pool = {
            let info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&descriptor_pool_sizes);
            unsafe { ctx.device().create_descriptor_pool(&info, None)? }
        };
        let descriptor_set = {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            unsafe { ctx.device().allocate_descriptor_sets(&info)? }
        }[0];

        // writes
        let buffer_infos = layout_buffers
            .iter()
            .map(|layout_buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(layout_buffer.buffer.buffer())
                    .offset(0)
                    .range(layout_buffer.buffer.size() as vk::DeviceSize)
            })
            .collect::<Vec<_>>();
        let writes = layout_buffers
            .iter()
            .zip(&buffer_infos)
            .map(|(layout_buffer, buffer_info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(layout_buffer.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info))
            })
            .collect::<Vec<_>>();
        unsafe {
            ctx.device().update_descriptor_sets(&writes, &[]);
        }

        Ok(Self {
            device: ctx.device(),
            descriptor_set_layout,
            descriptor_set,
            descriptor_pool,
        })
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for VkComputeShaderLayoutSet {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct VkComputeShaderLayoutBuffer {
    pub buffer: VkBuffer,
    pub binding: u32,
}
