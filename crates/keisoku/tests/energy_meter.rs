#[cfg(target_os = "macos")]
#[test]
fn energy_meter_measures_a_window() {
    let meter = keisoku::EnergyMeter::new();
    let window = meter.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reading = meter.stop(window).expect("energy reading on macOS");

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

#[cfg(target_os = "macos")]
#[test]
fn meter_survives_moving_between_threads() {
    let meter = keisoku::EnergyMeter::new();
    let (meter, window) = std::thread::spawn(move || {
        let window = meter.start();
        (meter, window)
    })
    .join()
    .expect("start thread");
    let reading = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        meter.stop(window)
    })
    .join()
    .expect("stop thread");
    let reading = reading.expect("energy reading after crossing threads");
    assert!(reading.energy.total().value() >= 0.0 && reading.energy.total().value().is_finite());
    assert!(reading.elapsed.value() >= 80);
}

#[cfg(target_os = "macos")]
#[test]
fn reused_windows_skip_the_subscription() {
    let meter = keisoku::EnergyMeter::new();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        let window = meter.start();
        let _ = meter.stop(window);
        let cycle = started.elapsed();
        println!("start/stop cycle {} ms", cycle.as_millis());
        assert!(
            cycle.as_millis() < 100,
            "a reused meter must not rebuild the IOReport subscription, cycle took {} ms",
            cycle.as_millis(),
        );
    }
}
