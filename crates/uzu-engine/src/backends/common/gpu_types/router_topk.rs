/// Maximum expert logits retained by the fused router/TopK kernel.
pub const ROUTER_TOPK_MAX_EXPERTS: u32 = 512;

/// Maximum selection width supported by the fused TopK router kernel.
pub const ROUTER_TOPK_MAX_SELECTED_EXPERTS: u32 = 128;

/// Maximum model width cached by the fused TopK router kernel.
pub const ROUTER_TOPK_MAX_MODEL_DIM: u32 = 4096;

/// Threads launched by one fused TopK router threadgroup.
pub const ROUTER_TOPK_THREADS_PER_THREADGROUP: u32 = 256;
