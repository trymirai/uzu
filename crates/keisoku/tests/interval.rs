#[cfg(target_os = "macos")]
use keisoku::{Energy, Interval, Power};

#[cfg(target_os = "macos")]
#[test]
fn interval_measures_a_window() {
    let mut meter = Interval::<(Energy, Power)>::new();
    let session = meter.begin();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let (energy, average_power) = meter.end(session);

    println!("energy {:.4} J  package {:.2} W", energy.total().value(), average_power.package.value(),);

    assert!(energy.total().value() >= 0.0 && energy.total().value().is_finite());
    assert!(average_power.total().value() >= 0.0);
}

#[cfg(target_os = "macos")]
#[test]
fn meter_survives_moving_between_threads() {
    let mut meter = Interval::<(Energy, Power)>::new();
    let (mut meter, session) = std::thread::spawn(move || {
        let session = meter.begin();
        (meter, session)
    })
    .join()
    .expect("start thread");
    let (energy, _average_power) = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        meter.end(session)
    })
    .join()
    .expect("stop thread");
    assert!(energy.total().value() >= 0.0 && energy.total().value().is_finite());
}

#[cfg(target_os = "macos")]
#[test]
fn reused_windows_skip_the_subscription() {
    let mut meter = Interval::<(Energy, Power)>::new();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        let session = meter.begin();
        let _ = meter.end(session);
        let cycle = started.elapsed();
        println!("begin/end cycle {} ms", cycle.as_millis());
        assert!(
            cycle.as_millis() < 100,
            "a reused meter must not rebuild the IOReport subscription, cycle took {} ms",
            cycle.as_millis(),
        );
    }
}
