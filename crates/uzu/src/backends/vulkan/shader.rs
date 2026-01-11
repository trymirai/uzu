use std::sync::Arc;
use ash::vk;

pub struct VkShader {
    device: Arc<ash::Device>,
    shader_module: vk::ShaderModule
}

impl VkShader {
    pub fn new(device: Arc<ash::Device>, file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let shader_module = {
            let shader_bytes = std::fs::read(file_path)?;
            let info = vk::ShaderModuleCreateInfo::default()
                .code(bytemuck::cast_slice(shader_bytes.as_slice()));
            unsafe { device.create_shader_module(&info, None)? }
        };
        
        Ok(Self {
            device: device.clone(),
            shader_module
        })
    }
    
    pub fn module(&self) -> vk::ShaderModule {
        self.shader_module
    }
}

impl Drop for VkShader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.shader_module, None);
        }
    }
}