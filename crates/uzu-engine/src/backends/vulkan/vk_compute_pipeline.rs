use std::{ffi::CString, sync::Arc};

use ash::{vk, vk::SpecializationInfo};

use super::{Error, VkContext};

pub struct VkComputePipeline {
    context: Arc<VkContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    push_constant_size: u32,
}
impl VkComputePipeline {
    pub fn new(
        ctx: Arc<VkContext>,
        shader_module: vk::ShaderModule,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_size: u32,
        entry_point: &str,
        specialization_info: &SpecializationInfo<'_>,
    ) -> Result<Self, Error> {
        let limit = ctx.physical_device().properties.limits.max_push_constants_size;
        if !push_constant_size.is_multiple_of(4) || push_constant_size > limit {
            return Err(Error::PushConstants {
                size: push_constant_size,
                limit,
            });
        }
        let entry_cstring = CString::new(entry_point)?;
        let pipeline_layout = {
            let ranges = (push_constant_size != 0)
                .then(|| {
                    vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_constant_size)
                })
                .into_iter()
                .collect::<Vec<_>>();
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(descriptor_set_layouts)
                .push_constant_ranges(&ranges);
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
            push_constant_size,
        })
    }

    pub fn context(&self) -> &Arc<VkContext> {
        &self.context
    }

    pub fn push_constant_size(&self) -> u32 {
        self.push_constant_size
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
