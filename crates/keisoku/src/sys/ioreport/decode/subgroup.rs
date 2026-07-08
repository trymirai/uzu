use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subgroup {
    CpuCorePerformanceStates,
    GpuPerformanceStates,
    DramBandwidth,
    Floor,
    Other,
}

impl Subgroup {
    pub(crate) fn classify(subgroup: &str) -> Subgroup {
        if subgroup == obfstr!("CPU Core Performance States") {
            Subgroup::CpuCorePerformanceStates
        } else if subgroup == obfstr!("GPU Performance States") {
            Subgroup::GpuPerformanceStates
        } else if subgroup == obfstr!("DRAM BW") {
            Subgroup::DramBandwidth
        } else if subgroup.contains(obfstr!("Floor")) {
            Subgroup::Floor
        } else {
            Subgroup::Other
        }
    }
}
