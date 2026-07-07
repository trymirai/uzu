use keisoku::{Bandwidth, CpuUsage, Energy, GpuUsage, IoReportGroups, Measured, NeuralEngine, Power};

#[test]
fn per_metric_groups() {
    assert_eq!(CpuUsage::GROUPS, IoReportGroups::CPU_STATS);
    assert_eq!(GpuUsage::GROUPS, IoReportGroups::GPU_STATS);
    assert_eq!(NeuralEngine::GROUPS, IoReportGroups::PMP);
    assert_eq!(Power::GROUPS, IoReportGroups::ENERGY_MODEL);
    assert_eq!(Energy::GROUPS, IoReportGroups::ENERGY_MODEL);
    assert_eq!(Bandwidth::GROUPS, IoReportGroups::AMC_STATS | IoReportGroups::PMP);
}

#[test]
fn tuple_folds_group_union() {
    assert_eq!(<(Energy, Power) as Measured>::GROUPS, IoReportGroups::ENERGY_MODEL);
    assert_eq!(
        <(CpuUsage, GpuUsage, NeuralEngine, Power, Bandwidth) as Measured>::GROUPS,
        IoReportGroups::ENERGY_MODEL
            | IoReportGroups::CPU_STATS
            | IoReportGroups::GPU_STATS
            | IoReportGroups::AMC_STATS
            | IoReportGroups::PMP,
    );
}
