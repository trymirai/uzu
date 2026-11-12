#[derive(PartialEq, Eq, Hash)]
pub struct VkComputePipelineKey {
    pub spirv: &'static [u8],
    pub entry_point: &'static str,
    pub specialization: Vec<(u32, u32)>,
    pub push_constant_size: u32,
    pub required_subgroup_size: Option<u32>,
}
