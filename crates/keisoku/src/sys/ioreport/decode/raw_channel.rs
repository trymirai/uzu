use super::{group_id::GroupId, residency_state::ResidencyState};

/// One decoded IOReport channel. Only the fields a channel's group actually
/// uses are populated (see `ioreport::channel`), so byte-counter channels carry
/// no residency states and energy channels carry no subgroup. `unit` is the raw
/// IOReport unit code (see `decode::unit`), populated only for energy channels.
#[derive(Default)]
pub struct RawChannel {
    pub(crate) group: GroupId,
    pub(crate) subgroup: String,
    pub(crate) name: String,
    pub(crate) unit: u64,
    pub(crate) integer_value: i64,
    pub(crate) states: Vec<ResidencyState>,
}
