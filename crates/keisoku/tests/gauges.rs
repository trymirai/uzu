#[cfg(target_os = "macos")]
#[test]
fn gauges_are_instantaneous() {
    use std::time::Instant;

    use keisoku::{Instant as Gauges, Memory, PackageWatts, TemperatureSensors};

    let mut gauges = Gauges::<(Memory, TemperatureSensors, PackageWatts)>::new();
    let started = Instant::now();
    let (memory, sensors, package) = gauges.read();
    let elapsed = started.elapsed();

    println!(
        "gauges in {} ms  memory {}  sensors {}  package {}",
        elapsed.as_millis(),
        memory.is_some(),
        sensors.len(),
        package.map_or("--".into(), |watts| format!("{:.2} W", watts.value())),
    );

    assert!(memory.is_some(), "sysinfo memory always readable on macOS");
    assert!(!sensors.is_empty(), "HID temperature sensors always present on macOS");
    assert!(
        elapsed.as_millis() < 300,
        "gauges must skip the IOReport subscription (~620 ms), took {} ms",
        elapsed.as_millis(),
    );
}
