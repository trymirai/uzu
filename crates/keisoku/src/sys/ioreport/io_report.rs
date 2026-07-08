use super::{
    IoReportFunctions, channel::for_each_channel, raw_energy_sample::RawEnergySample, subscription::Subscription,
};
use crate::sys::ioreport::{
    IoReportGroups,
    decode::{Channel, RawChannel},
};

pub(crate) struct IoReport {
    functions: &'static IoReportFunctions,
    subscription: Subscription,
}

impl IoReport {
    pub fn for_groups(groups: IoReportGroups) -> Option<Self> {
        let functions = IoReportFunctions::get()?;
        let subscription = Subscription::for_groups(groups, functions)?;
        Some(Self {
            functions,
            subscription,
        })
    }

    pub(crate) fn snapshot(&self) -> Option<RawEnergySample> {
        self.subscription.snapshot(self.functions).map(RawEnergySample)
    }

    pub(crate) fn for_each_channel(
        &self,
        begin: &RawEnergySample,
        end: &RawEnergySample,
        mut visit: impl FnMut(Channel, &RawChannel),
    ) {
        let Some(delta) = self.functions.create_samples_delta(&begin.0, &end.0) else {
            return;
        };
        for_each_channel(self.functions, &delta, &mut visit);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::IoReport;
    use crate::sys::ioreport::{IoReportGroups, decode::Channel};

    #[test]
    fn energy_channels_use_energy_quantity_units() {
        let Some(report) = IoReport::for_groups(IoReportGroups::ENERGY_MODEL) else {
            return;
        };
        let Some(begin) = report.snapshot() else {
            return;
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        let Some(end) = report.snapshot() else {
            return;
        };

        let mut energy_channels = 0u32;
        report.for_each_channel(
            &begin,
            &end,
            |channel, raw| {
                if let Channel::EnergyRail(_) = channel {
                    energy_channels += 1;
                    let quantity = (raw.unit >> 56) & 0xff;
                    assert_eq!(quantity, 3, "energy channel unit quantity should be Energy(3), got {quantity:#x}");
                    let exponent = ((raw.unit >> 32) & 0xff) as i32 - 127;
                    assert!(
                        (-12..=0).contains(&exponent),
                        "energy channel SI exponent {exponent} outside the expected pico..whole-joule range",
                    );
                }
            },
        );
        assert!(energy_channels > 0, "expected at least one Energy Model rail channel on macOS");
    }
}
