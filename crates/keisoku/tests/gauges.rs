#[cfg(target_os = "macos")]
#[test]
fn gauges_are_instantaneous() {
    use std::time::Instant;

    use keisoku::{Instant as Gauges, Memory, Select, TemperatureSensors};

    let mut gauges = Gauges::<Select![Memory, TemperatureSensors]>::new();
    let started = Instant::now();
    let sample = gauges.read();
    let memory = sample.get::<Memory>();
    let sensors = sample.get::<TemperatureSensors>();
    let elapsed = started.elapsed();

    println!("gauges in {} ms  memory {}  sensors {}", elapsed.as_millis(), memory.is_some(), sensors.len(),);

    assert!(memory.is_some(), "sysinfo memory always readable on macOS");
    assert!(!sensors.is_empty(), "HID temperature sensors always present on macOS");
    assert!(
        elapsed.as_millis() < 300,
        "gauges must skip the IOReport subscription (~620 ms), took {} ms",
        elapsed.as_millis(),
    );
}
