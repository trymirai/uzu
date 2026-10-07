use std::{ffi::CString, sync::Arc};

use ash::{vk, vk::SpecializationInfo};

use super::{Error, VkContext};

pub struct VkComputePipeline {
    context: Arc<VkContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}
impl VkComputePipeline {
    pub fn new(
        ctx: Arc<VkContext>,
        shader_module: vk::ShaderModule,
        descriptor_set_layout: vk::DescriptorSetLayout,
        entry_point: &str,
        specialization_info: &SpecializationInfo<'_>,
    ) -> Result<Self, Error> {
        let entry_cstring = CString::new(entry_point)?;
        let pipeline_layout = {
            let info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_set_layout));
            unsafe { ctx.device().create_pipeline_layout(&info, None)? }
        };

        let pipeline = {
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_cstring.as_c_str())
                .specialization_info(specialization_info);
            let info = vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);
            unsafe {
                ctx.device()
                    .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None)
                    .map_err(|(pipelines, error)| {
                        for pipeline in pipelines {
                            ctx.device().destroy_pipeline(pipeline, None);
                        }
                        ctx.device().destroy_pipeline_layout(pipeline_layout, None);
                        error
                    })?
            }
        }[0];

        Ok(Self {
            context: ctx,
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
            self.context.device().destroy_pipeline(self.pipeline, None);
            self.context.device().destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
