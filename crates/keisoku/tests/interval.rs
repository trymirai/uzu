#[cfg(target_os = "macos")]
use keisoku::{Energy, Interval, Power, Select};

#[cfg(target_os = "macos")]
#[test]
fn interval_measures_a_window() {
    let mut meter = Interval::<Select![Energy, Power]>::new();
    assert!(meter.is_available(), "macOS interval telemetry should report its source availability");
    let session = meter.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let sample = meter.stop(session);
    let energy = sample.get::<Energy>();
    let average_power = sample.get::<Power>();

    println!("energy {:.4} J  power {:.2} W", energy.total().value(), average_power.total().value());

    assert!(energy.total().value() >= 0.0 && energy.total().value().is_finite());
    assert!(average_power.total().value() >= 0.0);
}

#[cfg(target_os = "macos")]
#[test]
fn try_new_returns_available_meter() {
    let meter = Interval::<Select![Energy, Power]>::try_new();
    assert!(meter.as_ref().is_some_and(|meter| meter.is_available()));
}

#[cfg(target_os = "macos")]
#[test]
fn meter_survives_moving_between_threads() {
    let mut meter = Interval::<Select![Energy, Power]>::new();
    let (mut meter, session) = std::thread::spawn(move || {
        let session = meter.start();
        (meter, session)
    })
    .join()
    .expect("start thread");
    let energy = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        meter.stop(session).get::<Energy>().clone()
    })
    .join()
    .expect("stop thread");
    assert!(energy.total().value() >= 0.0 && energy.total().value().is_finite());
}

#[cfg(target_os = "macos")]
#[test]
fn reused_windows_skip_the_subscription() {
    let mut meter = Interval::<Select![Energy, Power]>::new();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        let session = meter.start();
        let _ = meter.stop(session);
        let cycle = started.elapsed();
        println!("begin/end cycle {} ms", cycle.as_millis());
        assert!(
            cycle.as_millis() < 100,
            "a reused meter must not rebuild the IOReport subscription, cycle took {} ms",
            cycle.as_millis(),
        );
    }
}
