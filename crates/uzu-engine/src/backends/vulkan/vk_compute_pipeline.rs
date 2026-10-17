use std::{ffi::CString, sync::Arc};

use ash::{vk, vk::SpecializationInfo};

use super::{Error, VkContext, VkDeviceCapabilities};

pub struct VkComputePipeline {
    context: Arc<VkContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}
impl VkComputePipeline {
    /// `required_subgroup_size` overrides the device's subgroup size policy, in full subgroups where enabled.
    pub fn new(
        ctx: Arc<VkContext>,
        shader_module: vk::ShaderModule,
        push_constant_size: u32,
        entry_point: &str,
        specialization_info: &SpecializationInfo<'_>,
        required_subgroup_size: Option<u32>,
    ) -> Result<Self, Error> {
        let entry_cstring = CString::new(entry_point)?;
        let pipeline_layout = {
            let ranges = (push_constant_size != 0)
                .then(|| {
                    vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_constant_size)
                })
                .into_iter()
                .collect::<Vec<_>>();
            let info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
            unsafe { ctx.device().create_pipeline_layout(&info, None)? }
        };

        let pipeline = {
            let mut subgroup_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
            let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_cstring.as_c_str())
                .specialization_info(specialization_info);
            let physical_device = ctx.physical_device();
            if let Some(size) = required_subgroup_size.or(physical_device.subgroup_properties.required_size) {
                subgroup_size.required_subgroup_size = size;
                stage_info = stage_info.push_next(&mut subgroup_size);
            }
            if required_subgroup_size.is_some()
                && physical_device.capabilities.contains(VkDeviceCapabilities::FULL_SUBGROUPS)
            {
                stage_info = stage_info.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
            }
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
