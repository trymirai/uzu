use std::{
    collections::HashMap,
    ffi::CString,
    hash::{BuildHasher, BuildHasherDefault, DefaultHasher},
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;
use parking_lot::Mutex;

use crate::backends::vulkan::{Error, VkContext, VkContextError, VkPipelineError};

const PIPELINE_CACHE_PREFIX: &str = "vulkan-pipeline-cache";

pub struct VkPipelines {
    device: ash::Device,
    pipelines: Mutex<HashMap<VkComputePipelineKey, Weak<VkComputePipeline>>>,
    cache: vk::PipelineCache,
    cache_file: Option<(PathBuf, u64)>,
}

impl VkPipelines {
    pub fn new(device: ash::Device) -> Result<Self, VkContextError> {
        let path = (!cfg!(test)).then(dirs::home_dir).flatten().map(|dir| {
            dir.join(".cache")
                .join("uzu")
                .join(format!("{PIPELINE_CACHE_PREFIX}-{}.bin", env!("UZU_VULKAN_SPIRV_HASH")))
        });
        let data = match path.as_ref().map(std::fs::read) {
            Some(Err(error)) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(VkContextError::PipelineCacheRead(path.unwrap_or_default(), error));
            },
            Some(Ok(data)) => data,
            _ => Vec::new(),
        };
        let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
        let cache =
            unsafe { device.create_pipeline_cache(&info, None) }.map_err(VkContextError::PipelineCacheCreate)?;
        let loaded_hash = BuildHasherDefault::<DefaultHasher>::default().hash_one(&data);
        Ok(Self {
            device,
            pipelines: Mutex::new(HashMap::new()),
            cache,
            cache_file: path.map(|path| (path, loaded_hash)),
        })
    }

    pub fn get(
        &self,
        context: &VkContext,
        key: VkComputePipelineKey,
    ) -> Result<Arc<VkComputePipeline>, Error> {
        let live = self.pipelines.lock().get(&key).and_then(Weak::upgrade);
        if let Some(pipeline) = live {
            return Ok(pipeline);
        }
        let pipeline = VkComputePipeline::new(context.arc(), &key, self.cache);
        if let Some(message) = context.take_validation_error() {
            return Err(Error::Validation(message));
        }
        let pipeline = Arc::new(pipeline?);
        let mut pipelines = self.pipelines.lock();
        pipelines.retain(|_, cached| cached.strong_count() > 0);
        Ok(match pipelines.get(&key).and_then(Weak::upgrade) {
            Some(winner) => winner,
            None => {
                pipelines.insert(key, Arc::downgrade(&pipeline));
                pipeline.clone()
            },
        })
    }

    fn save(&self) {
        static SAVES: AtomicUsize = AtomicUsize::new(0);
        let Some((path, loaded_hash)) = &self.cache_file else {
            return;
        };
        let Ok(data) = (unsafe { self.device.get_pipeline_cache_data(self.cache) }) else {
            return;
        };
        if BuildHasherDefault::<DefaultHasher>::default().hash_one(&data) == *loaded_hash {
            return;
        }
        let save = SAVES.fetch_add(1, Ordering::Relaxed);
        let temporary = path.with_extension(format!("{}-{save}.tmp", std::process::id()));
        let directory = path.parent().expect("pipeline cache paths have a parent directory");
        let saved = std::fs::create_dir_all(directory)
            .and_then(|()| std::fs::write(&temporary, &data))
            .and_then(|()| std::fs::rename(&temporary, path));
        if saved.is_err() {
            let _ = std::fs::remove_file(&temporary);
            return;
        }
        for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
            let other = entry.path();
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if other != *path && name.starts_with(PIPELINE_CACHE_PREFIX) && name.ends_with(".bin") {
                let _ = std::fs::remove_file(&other);
            }
        }
    }
}

impl Drop for VkPipelines {
    fn drop(&mut self) {
        self.save();
        unsafe { self.device.destroy_pipeline_cache(self.cache, None) };
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct VkComputePipelineKey {
    pub spirv: &'static [u8],
    pub entry_point: &'static str,
    pub specialization: Vec<(u32, u32)>,
    pub push_constant_size: u32,
    pub required_subgroup_size: Option<u32>,
}

pub struct VkComputePipeline {
    context: Arc<VkContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl VkComputePipeline {
    pub fn new(
        context: Arc<VkContext>,
        key: &VkComputePipelineKey,
        cache: vk::PipelineCache,
    ) -> Result<Self, VkPipelineError> {
        if let Some(size) = key.required_subgroup_size
            && !context.physical_device().limits.subgroup_size_requirable(size)
        {
            return Err(VkPipelineError::SubgroupSize(size));
        }
        let device = context.device();
        let entry_point = CString::new(key.entry_point).expect("entry points have no NUL");
        let words = ash::util::read_spv(&mut Cursor::new(key.spirv)).map_err(VkPipelineError::Spirv)?;
        let range = (key.push_constant_size != 0).then(|| {
            vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(key.push_constant_size)
        });
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(range.as_slice());
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
        let created = unsafe { device.create_compute_pipelines(cache, std::slice::from_ref(&info), None) };
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = created
            .map_err(|(_, error)| error)
            .inspect_err(|_| unsafe { device.destroy_pipeline_layout(pipeline_layout, None) })?[0];

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
