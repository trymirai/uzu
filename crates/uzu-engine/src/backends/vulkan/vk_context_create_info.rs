use super::{VkLogger, VkPrintlnLogger};

pub struct VkContextCreateInfo {
    pub with_validation: bool,
    pub logger: Box<dyn VkLogger>,
}
impl Default for VkContextCreateInfo {
    fn default() -> Self {
        Self {
            with_validation: true,
            logger: Box::new(VkPrintlnLogger),
        }
    }
}
