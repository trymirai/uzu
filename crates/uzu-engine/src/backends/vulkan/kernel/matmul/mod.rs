mod vk_gemv_tile;
mod vk_matmul_kernel;
mod vk_matmul_plan;

pub use vk_gemv_tile::VkGemvTile;
pub use vk_matmul_kernel::VkMatmulKernel;
pub use vk_matmul_plan::VkMatmulPlan;

use crate::backends::common::gpu_types::{gemm::GemmBPrologueKind, quantization::QuantizationMode};

/// B's prologue: its scheme, code mode, stored sign and group size.
type GemmBPrologue = (GemmBPrologueKind, QuantizationMode, bool, u32);
