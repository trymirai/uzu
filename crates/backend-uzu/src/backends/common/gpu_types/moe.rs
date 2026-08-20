/// Maximum expert rows cached by the fused router kernel.
pub const ROUTER_MAX_EXPERTS: u32 = 512;

/// Maximum TopK width supported by the fused router kernel.
pub const ROUTER_MAX_SELECTED_EXPERTS: u32 = 128;

/// Maximum model width cached by the fused router kernel.
pub const ROUTER_MAX_MODEL_DIM: u32 = 4096;

/// Threads launched by one fused router threadgroup.
pub const ROUTER_THREADS_PER_THREADGROUP: u32 = 256;
