use std::{ffi::CString, io::Cursor, sync::Arc};

use ash::vk;

use crate::backends::vulkan::{VkComputePipelineKey, VkContext, VkPipelineError};

pub struct VkComputePipeline {
    context: Arc<VkContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}
impl VkComputePipeline {
    pub fn new(
        context: Arc<VkContext>,
        key: &VkComputePipelineKey,
    ) -> Result<Self, VkPipelineError> {
        if let Some(size) = key.required_subgroup_size
            && !context.physical_device().subgroup_properties.requirable(size)
        {
            return Err(VkPipelineError::SubgroupSize(size));
        }
        let device = context.device();
        let entry_point = CString::new(key.entry_point)?;
        let words = ash::util::read_spv(&mut Cursor::new(key.spirv)).map_err(VkPipelineError::Spirv)?;
        let ranges = (key.push_constant_size != 0)
            .then(|| {
                vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(key.push_constant_size)
            })
            .into_iter()
            .collect::<Vec<_>>();
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = unsafe { device.create_shader_module(&module_info, None) }
            .inspect_err(|_| unsafe { device.destroy_pipeline_layout(pipeline_layout, None) })?;

        let entries = (0..)
            .zip(&key.specialization)
            .map(|(index, &(id, _))| vk::SpecializationMapEntry::default().constant_id(id).offset(4 * index).size(4))
            .collect::<Vec<_>>();
        let data = key.specialization.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let specialization = vk::SpecializationInfo::default().map_entries(&entries).data(bytemuck::cast_slice(&data));
        let mut subgroup_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry_point.as_c_str())
            .specialization_info(&specialization);
        if let Some(size) = key.required_subgroup_size {
            subgroup_size.required_subgroup_size = size;
            stage =
                stage.push_next(&mut subgroup_size).flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
        }
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
        let created =
            unsafe { device.create_compute_pipelines(context.pipeline_cache(), std::slice::from_ref(&info), None) };
        // The pipeline no longer needs its module.
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = match created {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, error)) => unsafe {
                for pipeline in pipelines {
                    device.destroy_pipeline(pipeline, None);
                }
                device.destroy_pipeline_layout(pipeline_layout, None);
                return Err(error.into());
            },
        };

        Ok(Self {
            context,
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
