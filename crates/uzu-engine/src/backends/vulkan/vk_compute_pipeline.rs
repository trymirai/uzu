use std::{
    collections::HashMap,
    ffi::CString,
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use ash::vk;
use parking_lot::Mutex;

use crate::backends::vulkan::{Error, VkDevice, VkPipelineError};

pub struct VkPipelines {
    device: Arc<VkDevice>,
    // Kept for the context's lifetime: kernels are created per step, and pipelines must not be compiled again.
    pipelines: Mutex<HashMap<VkComputePipelineKey, Arc<VkComputePipeline>>>,
    cache: vk::PipelineCache,
    cache_file: Option<VkPipelineCacheFile>,
    compiled_since_save: AtomicBool,
}

// A cache file that cannot be read or written only costs compiling the pipelines again, so its errors are ignored.
struct VkPipelineCacheFile {
    path: PathBuf,
    saved_size: Mutex<usize>,
}

impl VkPipelines {
    pub fn new(
        device: Arc<VkDevice>,
        cache_path: Option<PathBuf>,
    ) -> Result<Self, VkPipelineError> {
        let data = cache_path.as_ref().and_then(|path| std::fs::read(path).ok()).unwrap_or_default();
        let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
        let cache = unsafe { device.create_pipeline_cache(&info, None) }?;
        Ok(Self {
            device,
            pipelines: Mutex::new(HashMap::new()),
            cache,
            cache_file: cache_path.map(|path| VkPipelineCacheFile {
                path,
                saved_size: Mutex::new(data.len()),
            }),
            compiled_since_save: AtomicBool::new(false),
        })
    }

    pub fn get(
        &self,
        key: VkComputePipelineKey,
    ) -> Result<Arc<VkComputePipeline>, Error> {
        if let Some(pipeline) = self.pipelines.lock().get(&key) {
            return Ok(pipeline.clone());
        }
        // Compiled outside the lock; when two threads compile the same key, the first insert wins.
        let pipeline = Arc::new(VkComputePipeline::new(self.device.clone(), &key, self.cache)?);
        self.compiled_since_save.store(true, Ordering::Relaxed);
        Ok(self.pipelines.lock().entry(key).or_insert(pipeline).clone())
    }

    // Writes the cache file when pipelines were compiled since the last save, so a process that is killed later
    // still finds them next run. Pipelines found in the loaded cache do not grow it, so an unchanged size means there
    // is nothing new, and only the size is read from the driver.
    pub fn save(&self) {
        let Some(file) = &self.cache_file else {
            return;
        };
        if !self.compiled_since_save.swap(false, Ordering::Relaxed) {
            return;
        }
        let mut saved_size = file.saved_size.lock();
        let mut size = 0;
        let get_data = self.device.fp_v1_0().get_pipeline_cache_data;
        let queried = unsafe { get_data(self.device.handle(), self.cache, &mut size, std::ptr::null_mut()) };
        if queried != vk::Result::SUCCESS || size == *saved_size {
            return;
        }
        let Ok(data) = (unsafe { self.device.get_pipeline_cache_data(self.cache) }) else {
            return;
        };
        // Written beside the file and renamed over it, so no reader sees a partial cache; the temporary name is
        // unique per process and context, so concurrent saves never write the same file.
        let temporary = file.path.with_extension(format!("{}.{:p}.tmp", std::process::id(), self));
        let directory = file.path.parent().expect("pipeline cache paths have a parent directory");
        let saved = std::fs::create_dir_all(directory)
            .and_then(|()| std::fs::write(&temporary, &data))
            .and_then(|()| std::fs::rename(&temporary, &file.path));
        match saved {
            Ok(()) => *saved_size = data.len(),
            Err(_) => drop(std::fs::remove_file(&temporary)),
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
    device: Arc<VkDevice>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl VkComputePipeline {
    fn new(
        device: Arc<VkDevice>,
        key: &VkComputePipelineKey,
        cache: vk::PipelineCache,
    ) -> Result<Self, VkPipelineError> {
        if let Some(size) = key.required_subgroup_size
            && !device.physical_device.limits.subgroup_size_requirable(size)
        {
            return Err(VkPipelineError::SubgroupSize(size));
        }
        let entry_point = CString::new(key.entry_point).expect("entry points have no NUL");
        let words = ash::util::read_spv(&mut Cursor::new(key.spirv)).expect("the build script emits valid SPIR-V");
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
            device,
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
