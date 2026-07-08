use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct IntervalInputs: u16 {
        const ENERGY_RAILS = 1 << 0;
        const PACKAGE_WATTS = 1 << 1;
        const CPU_RESIDENCY = 1 << 2;
        const GPU_RESIDENCY = 1 << 3;
        const ANE_ACTIVITY = 1 << 4;
        const DRAM_BANDWIDTH = 1 << 5;
        const SOC_FREQUENCIES = 1 << 6;
    }
}

impl IntervalInputs {
    pub(crate) fn ioreport_groups(self) -> crate::providers::metric::IoReportGroups {
        use crate::providers::metric::IoReportGroups;

        let mut groups = IoReportGroups::empty();
        if self.contains(Self::ENERGY_RAILS) {
            groups |= IoReportGroups::ENERGY_MODEL;
        }
        if self.contains(Self::CPU_RESIDENCY) {
            groups |= IoReportGroups::CPU_STATS;
        }
        if self.contains(Self::GPU_RESIDENCY) {
            groups |= IoReportGroups::GPU_STATS;
        }
        if self.contains(Self::ANE_ACTIVITY) || self.contains(Self::DRAM_BANDWIDTH) {
            groups |= IoReportGroups::PMP;
        }
        if self.contains(Self::DRAM_BANDWIDTH) {
            groups |= IoReportGroups::AMC_STATS;
        }
        groups
    }

    pub(crate) fn needs_package_watts(self) -> bool {
        self.contains(Self::PACKAGE_WATTS)
    }

    pub(crate) fn needs_frequencies(self) -> bool {
        self.contains(Self::SOC_FREQUENCIES)
    }
}
