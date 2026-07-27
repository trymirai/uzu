use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GroupId {
    EnergyModel,
    AmcStats,
    Pmp,
    #[default]
    Other,
}

impl GroupId {
    pub(crate) fn classify(group: &str) -> GroupId {
        if group == obfstr!("Energy Model") {
            GroupId::EnergyModel
        } else if group == obfstr!("AMC Stats") {
            GroupId::AmcStats
        } else if group == obfstr!("PMP") {
            GroupId::Pmp
        } else {
            GroupId::Other
        }
    }
}
