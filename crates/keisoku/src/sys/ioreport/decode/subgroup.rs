use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subgroup {
    DcsBandwidth,
    DramBandwidth,
    Floor,
    Other,
}

impl Subgroup {
    pub(crate) fn classify(subgroup: &str) -> Subgroup {
        if subgroup == obfstr!("DCS BW") {
            Subgroup::DcsBandwidth
        } else if subgroup == obfstr!("DRAM BW") {
            Subgroup::DramBandwidth
        } else if subgroup.contains(obfstr!("Floor")) {
            Subgroup::Floor
        } else {
            Subgroup::Other
        }
    }
}
