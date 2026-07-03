use std::time::Duration;

use keisoku::{Collector, SensorKind, sensors, sensors_available};

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

#[test]
fn one_snapshot_is_well_formed() {
    let mut collector = Collector::new();
    let snapshot = collector.sample(Duration::from_millis(120));
    if let Some(power) = &snapshot.power {
        assert!(power.total().value().is_finite() && power.total().value() >= 0.0);
    }
    if let Some(memory) = &snapshot.memory {
        assert!(memory.ram_total >= memory.ram_usage);
    }
}
