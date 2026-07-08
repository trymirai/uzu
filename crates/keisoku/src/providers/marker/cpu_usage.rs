use super::{interval_set::IntervalMetric, typelist::Metric};
use crate::{
    providers::data::CpuMetrics,
    sources::interval::{IntervalFrame, IntervalInputs},
    sys::ioreport::decode::{self, CpuResidency},
    units::{Megahertz, Percent},
};

pub struct CpuUsage;

impl Metric for CpuUsage {
    type Value = CpuMetrics;
    const TYPE_BIT: u128 = 1 << 17;
}

impl IntervalMetric for CpuUsage {
    const INPUTS: IntervalInputs = IntervalInputs::CPU_RESIDENCY.union(IntervalInputs::SOC_FREQUENCIES);

    fn finish(frame: &IntervalFrame<'_>) -> CpuMetrics {
        let acc = frame.cpu.as_ref().cloned().unwrap_or_default();
        let frequencies = frame.frequencies.as_ref().cloned().unwrap_or_default();
        project_cpu(acc, frequencies)
    }
}

fn project_cpu(
    acc: CpuResidency,
    ctx: decode::FrequencyTables<'_>,
) -> CpuMetrics {
    let ecpu_readings: Vec<(u32, f32)> = acc.ecpu.iter().copied().filter(|&(_, percent)| percent > 0.0).collect();
    let ecpu = decode::average_cluster_frequency(&ecpu_readings, ctx.ecpu);
    let pcpu = decode::average_cluster_frequency(&acc.pcpu, ctx.pcpu);
    let efficiency_cores = ctx.ecpu_cores as f32;
    let performance_cores = ctx.pcpu_cores as f32;
    let usage = decode::divide_or_zero(
        ecpu.1 * efficiency_cores + pcpu.1 * performance_cores,
        efficiency_cores + performance_cores,
    );
    CpuMetrics {
        usage: Percent(usage * 100.0),
        ecpu_frequency: Megahertz(ecpu.0),
        pcpu_frequency: Megahertz(pcpu.0),
    }
}
