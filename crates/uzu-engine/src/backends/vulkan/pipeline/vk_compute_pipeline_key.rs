/// The full recipe of a generated compute pipeline, compared by content: equal keys create interchangeable pipelines.
#[derive(PartialEq, Eq, Hash)]
pub struct VkComputePipelineKey {
    pub spirv: &'static [u8],
    pub entry_point: &'static str,
    /// `(constant_id, value)` of each specialization constant, a 4-byte word.
    pub specialization: Vec<(u32, u32)>,
    pub push_constant_size: u32,
    /// Pins the pipeline's subgroups to this size, every one full.
    pub required_subgroup_size: Option<u32>,
}
