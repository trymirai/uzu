use keisoku::{SensorKind, sensors, sensors_available};

#[test]
fn sensor_queries_do_not_panic() {
    let _ = sensors_available();
    for kind in [SensorKind::Temperature, SensorKind::Voltage, SensorKind::Current] {
        for sensor in sensors(kind) {
            assert_eq!(sensor.kind, kind);
            assert!(sensor.value.is_finite());
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn one_snapshot_is_well_formed() {
    use keisoku::{Instant, Interval, Memory, Power};

    let mut soc = Interval::<Power>::new();
    let session = soc.begin();
    std::thread::sleep(std::time::Duration::from_millis(120));
    let power = soc.end(session);
    assert!(power.total().value().is_finite() && power.total().value() >= 0.0);

    let mut gauges = Instant::<Memory>::new();
    if let Some(memory) = gauges.read() {
        assert!(memory.ram_total >= memory.ram_usage);
    }
}
