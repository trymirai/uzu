use bitflags::bitflags;

use crate::sys::ioreport::{IoReportGroups, decode::Channel};

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct IntervalInputs: u16 {
        const ENERGY_RAILS = 1 << 0;
        const CPU_RESIDENCY = 1 << 1;
        const GPU_RESIDENCY = 1 << 2;
        const ANE_ACTIVITY = 1 << 3;
        const DRAM_BANDWIDTH = 1 << 4;
        const SOC_FREQUENCIES = 1 << 5;
    }
}

impl IntervalInputs {
    pub(crate) fn ioreport_groups(self) -> IoReportGroups {
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
        if self.contains(Self::ANE_ACTIVITY) {
            groups |= IoReportGroups::PMP;
        }
        if self.contains(Self::DRAM_BANDWIDTH) {
            groups |= IoReportGroups::AMC_STATS | IoReportGroups::PMP;
        }
        groups
    }

    pub(crate) fn wants(
        self,
        channel: Channel,
    ) -> bool {
        let flag = match channel {
            Channel::EnergyRail(_) => Self::ENERGY_RAILS,
            Channel::CpuCluster(_) => Self::CPU_RESIDENCY,
            Channel::GpuState => Self::GPU_RESIDENCY,
            Channel::AneBandwidth => Self::ANE_ACTIVITY,
            Channel::DramBytes(_) | Channel::DramHistogram(_) => Self::DRAM_BANDWIDTH,
        };
        self.contains(flag)
    }

    pub(crate) fn needs_frequencies(self) -> bool {
        self.contains(Self::SOC_FREQUENCIES)
    }
}
