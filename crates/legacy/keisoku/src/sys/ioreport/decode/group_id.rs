use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupId {
    EnergyModel,
    AmcStats,
    Pmp,
    #[default]
    Other,
}

impl GroupId {
    pub fn classify(group: &str) -> GroupId {
        if group == obfstr!("Energy Model") {
            GroupId::EnergyModel
        } else if group == obfstr!("AMC Stats") {
            GroupId::AmcStats
        } else if group == obfstr!("PMP") || group == obfstr!("PMP0") || group == obfstr!("PMP1") {
            GroupId::Pmp
        } else {
            GroupId::Other
        }
    }
}
