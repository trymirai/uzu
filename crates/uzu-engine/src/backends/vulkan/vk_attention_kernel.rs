use super::Vulkan;
use crate::backends::common::kernel::attention::SinglePassAttention;

pub type VkAttentionKernel = SinglePassAttention<Vulkan>;
