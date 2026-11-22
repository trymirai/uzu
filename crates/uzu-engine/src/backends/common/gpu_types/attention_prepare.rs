use bytemuck::NoUninit;

/// Parameters of attention with separate key and value inputs, L2 clipping or log scaling of its heads.
#[derive(Clone, Copy, Debug, Default, NoUninit)]
#[repr(C)]
pub struct AttentionPrepareParams {
    pub key_row_stride: u32,
    pub value_row_stride: u32,
    /// The L2 norms that query and key heads are clipped to.
    pub query_clip_threshold: f32,
    pub key_clip_threshold: f32,
    /// Queries are scaled by `1 + alpha * ln(max((position + 1) / n_floor, 1))`.
    pub log_scaling_n_floor: f32,
    pub log_scaling_alpha: f32,
    /// The position of the first suffix token; a trie node adds its height.
    pub position_offset: u32,
}

/// Parameters of `AttentionPrepareSeparate`, attention whose queries, keys and values project separate inputs.
#[derive(Clone, Copy, Debug, Default, NoUninit)]
#[repr(C)]
pub struct SeparatePrepareParams {
    pub query_row_stride: u32,
    pub key_row_stride: u32,
    pub value_row_stride: u32,
    /// The element offset of the relative states in a query row.
    pub relative_offset: u32,
    /// The cache slot of the first suffix token.
    pub kv_token_offset: u32,
    /// Query rows per key-value head: the grouped heads times the suffix length.
    pub query_rows: u32,
    /// The position of the first suffix token; a trie node adds its height.
    pub position_offset: u32,
    pub query_clip_threshold: f32,
    pub key_clip_threshold: f32,
    pub relative_clip_threshold: f32,
    /// Queries and relative states are scaled by `1 + alpha * ln(max((position + 1) / n_floor, 1))`.
    pub log_scaling_n_floor: f32,
    pub log_scaling_alpha: f32,
    pub key_conv_scale: f32,
    pub value_conv_scale: f32,
    pub value_norm_epsilon: f32,
}
