use ash::vk;

/// The full recipe of a generated compute pipeline, compared by content: equal keys create interchangeable pipelines.
#[derive(PartialEq, Eq, Hash)]
pub struct VkComputePipelineKey {
    spirv: &'static [u8],
    entry_point: &'static str,
    /// `(constant_id, offset, size)` of each specialization map entry.
    specialization_entries: Vec<(u32, u32, usize)>,
    specialization_data: Vec<u8>,
    push_constant_size: u32,
    required_subgroup_size: Option<u32>,
}
impl VkComputePipelineKey {
    pub fn new(
        spirv: &'static [u8],
        entry_point: &'static str,
        specialization_entries: &[vk::SpecializationMapEntry],
        specialization_data: &[u8],
        push_constant_size: u32,
        required_subgroup_size: Option<u32>,
    ) -> Self {
        Self {
            spirv,
            entry_point,
            specialization_entries: specialization_entries
                .iter()
                .map(|entry| (entry.constant_id, entry.offset, entry.size))
                .collect(),
            specialization_data: specialization_data.to_vec(),
            push_constant_size,
            required_subgroup_size,
        }
    }
}
