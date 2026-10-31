/// The key head dimension every GDN kernel takes.
pub const DELTA_NET_HEAD_K_DIM: u32 = 128;
/// The value heads of a key head that the chunked precompute shares a workgroup among, at most.
pub const DELTA_NET_CHUNKED_MAX_GROUPS: u32 = 4;
/// GDN tree verification's token block: trees of up to one or two.
pub const TREE_VERIFY_TOKEN_BLOCK: u32 = 16;
