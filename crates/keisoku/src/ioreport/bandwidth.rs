use obfstr::obfstr;

use super::{
    SocSample,
    channel::{ResidencyState, residency_active_percent, residency_weighted_gbps},
};

#[derive(Clone, Copy)]
enum Flow {
    Read,
    Write,
}

#[derive(Clone, Copy)]
enum Subsystem {
    Dram,
    NeuralEngine,
}

#[derive(Default)]
pub(super) struct BandwidthAccumulator {
    dram_read_bytes: f64,
    dram_write_bytes: f64,
    ane_read_bytes: f64,
    ane_write_bytes: f64,
    dram_read_histogram_gbps: f32,
    dram_write_histogram_gbps: f32,
    ane_read_histogram_gbps: f32,
    ane_write_histogram_gbps: f32,
}

impl BandwidthAccumulator {
    fn add_dram_bytes(
        &mut self,
        flow: Flow,
        bytes: f64,
    ) {
        match flow {
            Flow::Read => self.dram_read_bytes += bytes,
            Flow::Write => self.dram_write_bytes += bytes,
        }
    }

    fn add_ane_bytes(
        &mut self,
        flow: Flow,
        bytes: f64,
    ) {
        match flow {
            Flow::Read => self.ane_read_bytes += bytes,
            Flow::Write => self.ane_write_bytes += bytes,
        }
    }

    pub(super) fn finish(
        self,
        window_milliseconds: u64,
        result: &mut SocSample,
    ) {
        let window_seconds = (window_milliseconds as f64 / 1000.0).max(0.001);
        let to_gbps = |bytes: f64| (bytes / window_seconds / 1e9) as f32;

        result.dram_read_gbps = if self.dram_read_bytes > 0.0 {
            to_gbps(self.dram_read_bytes)
        } else {
            self.dram_read_histogram_gbps
        };
        result.dram_write_gbps = if self.dram_write_bytes > 0.0 {
            to_gbps(self.dram_write_bytes)
        } else {
            self.dram_write_histogram_gbps
        };
        result.ane_read_gbps = if self.ane_read_bytes > 0.0 {
            to_gbps(self.ane_read_bytes)
        } else {
            self.ane_read_histogram_gbps
        };
        result.ane_write_gbps = if self.ane_write_bytes > 0.0 {
            to_gbps(self.ane_write_bytes)
        } else {
            self.ane_write_histogram_gbps
        };
    }
}

pub(super) fn accumulate_amc_bandwidth(
    bytes: i64,
    channel: &str,
    bandwidth: &mut BandwidthAccumulator,
) {
    let aggregate = strip_die_prefix(channel);
    let target = if aggregate == obfstr!("DCS RD") {
        (Subsystem::Dram, Flow::Read)
    } else if aggregate == obfstr!("DCS WR") {
        (Subsystem::Dram, Flow::Write)
    } else if aggregate == obfstr!("ANE DCS RD") {
        (Subsystem::NeuralEngine, Flow::Read)
    } else if aggregate == obfstr!("ANE DCS WR") {
        (Subsystem::NeuralEngine, Flow::Write)
    } else {
        return;
    };
    let bytes = bytes as f64;
    if bytes <= 0.0 {
        return;
    }
    match target.0 {
        Subsystem::Dram => bandwidth.add_dram_bytes(target.1, bytes),
        Subsystem::NeuralEngine => bandwidth.add_ane_bytes(target.1, bytes),
    }
}

pub(super) fn accumulate_pmp(
    states: &[ResidencyState],
    subgroup: &str,
    channel: &str,
    bandwidth: &mut BandwidthAccumulator,
    result: &mut SocSample,
) {
    if subgroup.contains(obfstr!("Floor")) && (channel == obfstr!("ANE-AF-BW") || channel == obfstr!("ANE-DCS-BW")) {
        result.ane_active_percent = result.ane_active_percent.max(residency_active_percent(states));
    } else if subgroup == obfstr!("AF BW") {
        if channel == obfstr!("ANE0 RD") {
            bandwidth.ane_read_histogram_gbps = bandwidth.ane_read_histogram_gbps.max(residency_weighted_gbps(states));
        } else if channel == obfstr!("ANE0 WR") {
            bandwidth.ane_write_histogram_gbps =
                bandwidth.ane_write_histogram_gbps.max(residency_weighted_gbps(states));
        }
    } else if subgroup == obfstr!("DRAM BW") {
        let gbps = residency_weighted_gbps(states);
        match flow(channel) {
            Some(Flow::Read) => bandwidth.dram_read_histogram_gbps = bandwidth.dram_read_histogram_gbps.max(gbps),
            Some(Flow::Write) => bandwidth.dram_write_histogram_gbps = bandwidth.dram_write_histogram_gbps.max(gbps),
            None => {},
        }
    }
}

fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

fn flow(channel: &str) -> Option<Flow> {
    if channel.contains(obfstr!("RD+WR")) || channel.ends_with(obfstr!("RW")) {
        None
    } else if channel.contains(obfstr!("WR")) {
        Some(Flow::Write)
    } else if channel.contains(obfstr!("RD")) {
        Some(Flow::Read)
    } else {
        None
    }
}
