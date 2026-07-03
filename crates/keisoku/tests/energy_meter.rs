#[cfg(target_os = "macos")]
#[test]
fn energy_meter_measures_a_window() {
    let meter = keisoku::EnergyMeter::start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reading = meter.stop().expect("energy reading on macOS");

    println!(
        "energy {:.4} J  package {:.2} W  elapsed {} ms  from_smc {}",
        reading.energy.total().value(),
        reading.average_power.package.value(),
        reading.elapsed.value(),
        reading.package_from_smc,
    );

    assert!(reading.energy.total().value() >= 0.0 && reading.energy.total().value().is_finite());
    assert!(reading.average_power.total().value() >= 0.0);
    assert!(reading.elapsed.value() >= 150);
}
