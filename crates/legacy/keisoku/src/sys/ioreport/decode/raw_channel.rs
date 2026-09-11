use super::{group_id::GroupId, residency_state::ResidencyState};

#[derive(Default)]
pub struct RawChannel {
    pub group: GroupId,
    pub subgroup: String,
    pub name: String,
    pub unit: u64,
    pub integer_value: i64,
    pub states: Vec<ResidencyState>,
}
