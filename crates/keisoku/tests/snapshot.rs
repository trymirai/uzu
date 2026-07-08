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
    use keisoku::{Instant, Interval, Memory, Power, Select};

    let mut soc = Interval::<Select![Power]>::new();
    let session = soc.start();
    std::thread::sleep(std::time::Duration::from_millis(120));
    let sample = soc.stop(session);
    let power = sample.get::<Power>();
    assert!(power.total().value().is_finite() && power.total().value() >= 0.0);

    let mut gauges = Instant::<Select![Memory]>::new();
    let sample = gauges.read();
    if let Some(memory) = sample.get::<Memory>() {
        assert!(memory.ram_total >= memory.ram_usage);
    }
}
