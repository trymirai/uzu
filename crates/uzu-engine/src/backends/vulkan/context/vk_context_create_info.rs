use std::path::PathBuf;

pub struct VkContextCreateInfo {
    pub with_validation: bool,
    /// Where compiled pipelines persist between runs; `None` compiles each one again in every process.
    pub pipeline_cache_path: Option<PathBuf>,
}
impl Default for VkContextCreateInfo {
    fn default() -> Self {
        Self {
            // Android loads layers only from an app's own package or on debuggable builds.
            with_validation: !cfg!(target_os = "android"),
            pipeline_cache_path: dirs::cache_dir().map(|dir| dir.join("uzu").join("vulkan-pipeline-cache.bin")),
        }
    }
}
