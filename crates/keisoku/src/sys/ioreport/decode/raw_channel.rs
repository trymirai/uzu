use super::{group_id::GroupId, residency_state::ResidencyState};

#[derive(Default)]
pub struct RawChannel {
    pub(crate) group: GroupId,
    pub(crate) subgroup: String,
    pub(crate) name: String,
    pub(crate) unit: u64,
    pub(crate) integer_value: i64,
    pub(crate) states: Vec<ResidencyState>,
}
