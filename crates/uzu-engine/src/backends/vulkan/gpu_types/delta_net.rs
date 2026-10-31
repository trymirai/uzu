/// The key head dimension every GDN kernel takes.
pub const DELTA_NET_HEAD_K_DIM: u32 = 128;
/// The chunked GDN prefill's token chunk, common `delta_net_chunked_prefill::CHUNK_SIZE`.
pub const DELTA_NET_CHUNKED_CHUNK: u32 = 64;
/// The causal token block: the chunked precompute writes T and qk_scaled up to the end of each row's block, where the
/// chunked output reads them.
pub const DELTA_NET_CHUNKED_BLOCK: u32 = 16;
/// The value heads of a key head that the chunked precompute shares a workgroup among, at most.
pub const DELTA_NET_CHUNKED_MAX_GROUPS: u32 = 4;
/// GDN tree verification's token block: trees of up to one or two.
pub const TREE_VERIFY_TOKEN_BLOCK: u32 = 16;
