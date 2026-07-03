#[cfg(target_os = "macos")]
#[test]
fn gauges_are_instantaneous() {
    use std::time::Instant;

    let mut collector = keisoku::Collector::new();
    let started = Instant::now();
    let gauges = collector.gauges();
    let elapsed = started.elapsed();

    println!(
        "gauges in {} ms  memory {}  sensors {}  package {}",
        elapsed.as_millis(),
        gauges.memory.is_some(),
        gauges.sensors.len(),
        gauges.package_watts.map_or("--".into(), |w| format!("{:.2} W", w.value())),
    );

    assert!(gauges.memory.is_some(), "sysinfo memory always readable on macOS");
    assert!(!gauges.sensors.is_empty(), "HID temperature sensors always present on macOS");
    assert!(
        elapsed.as_millis() < 300,
        "gauges must skip the IOReport subscription (~620 ms), took {} ms",
        elapsed.as_millis(),
    );
}
