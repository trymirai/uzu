pub mod config;
pub mod image_processing;
pub mod tower;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_tower;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod lfm3_metal_tower;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod common;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod pipeline;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod any_pipeline;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use any_pipeline::{AnyPreparedInput, AnyVlmPipeline};

pub use config::{CanvasConfig, VisionConfig, VisionModelConfig};
pub use image_processing::{
    IMAGE_PAD, PreprocessedVision, build_placeholder_tokens_with_config, compute_canvas_mrope_with_config,
    preprocess_image,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_tower::MetalVisionTower;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use lfm3_metal_tower::Lfm3MetalVisionTower;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use lfm3_pipeline::{Lfm3Image, Lfm3PreparedInput, Lfm3VlPipeline};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use common::{
    MultimodalMessage, MultimodalToolCall, StreamingTokenDecoder, VlmSampler, VlmSamplingConfig,
    VlmState, hash_image, render_assistant_text, render_tool_response_text, tools_system_preamble,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use pipeline::{PreparedMultimodalInput, VlmPipeline};
pub use tower::VisionTower;
pub mod lfm3_image_processing;
pub mod lfm3_pipeline;
pub mod lfm3_tower;
