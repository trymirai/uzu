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
