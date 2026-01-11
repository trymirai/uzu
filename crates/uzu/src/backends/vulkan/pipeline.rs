use std::ffi::CStr;
use std::sync::Arc;
use ash::vk;
use crate::backends::vulkan::context::VkContext;

const MAIN: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

pub struct VkComputePipeline {
    device: Arc<ash::Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl VkComputePipeline {
    pub fn new(
        ctx: &VkContext,
        shader_module: vk::ShaderModule,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            unsafe { ctx.device().create_pipeline_layout(&info, None)? }
        };
        
        let pipeline = {
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(MAIN);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);
            unsafe {
                ctx.device().create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&info),
                    None
                ).map_err(|(_, err)| err)?
            }
        }[0];

        Ok(Self {
            device: ctx.device(),
            pipeline,
            pipeline_layout,
        })
    }
    
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
    
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for VkComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}