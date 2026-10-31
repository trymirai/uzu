pub const MODEL_MAX: u32 = 64;
pub const POOL_HIDDEN_MAX: u32 = 16;
pub const EDGE_HIDDEN_MAX: u32 = 128;
pub const CHILDREN_MAX: u32 = 4;
pub const SLOT_NUMBERS: u32 = 4;
pub const POOL_FEATURES: u32 = 3;
pub const POOL_INPUTS: u32 = 3;
pub const RELATIONS: u32 = 6;
pub const TREE_ROWS_MAX: u32 = 1024;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ConfidenceHeadParams {
    pub candidates: u32,
    pub children: u32,
    pub tree_slot_count: u32,
    pub vocab_size: u32,
    pub logit_cap: f32,
    pub epsilon: f32,
}
