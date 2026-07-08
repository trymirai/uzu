use crate::{
    Select,
    providers::metric::{
        Bandwidth, Battery, CpuUsage, CurrentSensors, Energy, Fans, GpuUsage, InstantSet, IntervalMetric, IntervalSet,
        Memory, Power, RailPower, TemperatureSensors, Temps, Thermal, VoltageSensors,
    },
};

#[test]
fn interval_inputs_union_matches_selected_metrics() {
    type Meter = Select![Energy, Power, CpuUsage];
    assert!(Meter::INPUTS.contains(Energy::INPUTS));
    assert!(Meter::INPUTS.contains(Power::INPUTS));
    assert!(Meter::INPUTS.contains(CpuUsage::INPUTS));
}

#[test]
fn metric_inputs_are_domain_specific() {
    assert!(Energy::INPUTS.contains(Energy::INPUTS));
    assert!(Power::INPUTS.contains(Power::INPUTS));
    assert!(CpuUsage::INPUTS.contains(CpuUsage::INPUTS));
    assert!(GpuUsage::INPUTS.contains(GpuUsage::INPUTS));
    assert!(Bandwidth::INPUTS.contains(Bandwidth::INPUTS));
}

#[test]
fn recursive_select_is_not_limited_to_eight_metrics() {
    fn assert_instant_set<T: InstantSet>() {}

    assert_instant_set::<
        Select![Memory, Fans, Battery, Temps, Thermal, TemperatureSensors, VoltageSensors, CurrentSensors, RailPower,],
    >();
}
