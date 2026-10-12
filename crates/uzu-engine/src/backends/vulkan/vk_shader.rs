use std::{io::Cursor, sync::Arc};

use ash::vk;

use super::{Error, VkContext};

pub struct VkShader {
    context: Arc<VkContext>,
    shader_module: vk::ShaderModule,
}

impl VkShader {
    pub fn new(
        context: Arc<VkContext>,
        spirv: &[u8],
    ) -> Result<Self, Error> {
        let words = ash::util::read_spv(&mut Cursor::new(spirv))?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        let shader_module = unsafe { context.device().create_shader_module(&info, None)? };
        Ok(Self {
            context,
            shader_module,
        })
    }

    pub fn module(&self) -> vk::ShaderModule {
        self.shader_module
    }
}

impl Drop for VkShader {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_shader_module(self.shader_module, None);
        }
    }
}
