/// AttentionGemm's query tile: a workgroup's query rows of one key-value head.
pub const ATTENTION_GEMM_QUERIES: u32 = 32;
/// AttentionGemm's key blocks, which a workgroup stages in turn.
pub const ATTENTION_GEMM_BLOCK_KEYS: u32 = 32;
/// The workgroup memory AttentionGemm's query, key or value, weight and scale tiles take.
pub const ATTENTION_GEMM_SHARED_BYTES: u32 = 29312;
