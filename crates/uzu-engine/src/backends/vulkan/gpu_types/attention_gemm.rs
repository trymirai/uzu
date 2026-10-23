/// AttentionGemm's query tile: a workgroup's queries of one head.
pub const ATTENTION_GEMM_QUERIES: u32 = 32;
/// The workgroup memory AttentionGemm's query, key or value, and weight tiles take.
pub const ATTENTION_GEMM_SHARED_BYTES: u32 = 29696;
