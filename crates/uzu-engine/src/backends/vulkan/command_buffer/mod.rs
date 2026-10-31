mod error;
mod vk_command_buffer;
mod vk_command_buffer_completed;
mod vk_command_buffer_encoding;
mod vk_command_buffer_executable;
mod vk_command_buffer_pending;
mod vk_command_buffer_resources;
mod vk_command_buffer_submission;
mod vk_debug_group_span;
mod vk_timestamp_recorder;

pub use error::VkCommandBufferError;
pub use vk_command_buffer::VkCommandBuffer;
pub use vk_command_buffer_completed::VkCommandBufferCompleted;
pub use vk_command_buffer_encoding::VkCommandBufferEncoding;
pub use vk_command_buffer_executable::VkCommandBufferExecutable;
pub use vk_command_buffer_pending::VkCommandBufferPending;
pub use vk_command_buffer_resources::VkCommandBufferResources;
pub use vk_command_buffer_submission::VkCommandBufferSubmission;
pub use vk_debug_group_span::VkDebugGroupSpan;
pub use vk_timestamp_recorder::VkTimestampRecorder;

/// A completed segment's GPU time and span timestamps.
pub type VkSegmentResult = Result<(std::time::Duration, Box<[u64]>), crate::backends::vulkan::Error>;
