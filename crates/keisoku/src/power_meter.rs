use crate::units::{Joules, Watts};

pub struct PowerReading {
    pub cpu: Option<Watts>,
    pub gpu: Option<Watts>,
    pub ane: Option<Watts>,
    pub ram: Option<Watts>,
    pub total: Watts,
    pub energy: Joules,
    pub samples: u64,
}

pub struct PowerMeter {
    inner: inner::Inner,
}

impl PowerMeter {
    pub fn new() -> Self {
        Self {
            inner: inner::Inner::new(),
        }
    }

    pub fn start(&mut self) {
        self.inner.start();
    }

    pub fn stop(&mut self) -> Option<PowerReading> {
        self.inner.stop()
    }
}

impl Default for PowerMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "macos")]
mod inner {
    use super::PowerReading;
    use crate::{
        Select,
        providers::{
            Interval, Session,
            marker::{Energy, Power},
        },
    };

    type Metrics = Select![Energy, Power];

    pub(super) struct Inner {
        meter: Interval<Metrics>,
        session: Option<Session<Metrics>>,
    }

    impl Inner {
        pub(super) fn new() -> Self {
            Self {
                meter: Interval::new(),
                session: None,
            }
        }

        pub(super) fn start(&mut self) {
            if self.meter.is_available() {
                self.session = Some(self.meter.start());
            }
        }

        pub(super) fn stop(&mut self) -> Option<PowerReading> {
            let session = self.session.take()?;
            let sample = self.meter.stop(session);
            let power = sample.get::<Power>();
            let energy = sample.get::<Energy>();
            Some(PowerReading {
                cpu: Some(power.cpu),
                gpu: Some(power.gpu),
                ane: Some(power.ane),
                ram: Some(power.ram),
                total: power.total(),
                energy: energy.total(),
                samples: 1,
            })
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod inner {
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use super::PowerReading;
    use crate::{
        Select,
        providers::{Instant as InstantMeter, marker::RailPower},
        units::{Joules, Watts},
    };

    const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

    #[derive(Default)]
    struct Accumulator {
        energy_joules: f64,
        elapsed_seconds: f64,
        samples: u64,
    }

    pub(super) struct Inner {
        running: Arc<AtomicBool>,
        accumulator: Arc<Mutex<Accumulator>>,
        worker: Option<JoinHandle<()>>,
    }

    impl Inner {
        pub(super) fn new() -> Self {
            Self {
                running: Arc::new(AtomicBool::new(false)),
                accumulator: Arc::new(Mutex::new(Accumulator::default())),
                worker: None,
            }
        }

        pub(super) fn start(&mut self) {
            *self.accumulator.lock().unwrap() = Accumulator::default();
            self.running.store(true, Ordering::SeqCst);
            let running = self.running.clone();
            let accumulator = self.accumulator.clone();
            self.worker = Some(thread::spawn(move || {
                let mut meter = InstantMeter::<Select![RailPower]>::new();
                let mut last = Instant::now();
                while running.load(Ordering::SeqCst) {
                    thread::sleep(SAMPLE_INTERVAL);
                    let now = Instant::now();
                    let seconds = now.duration_since(last).as_secs_f64();
                    last = now;
                    let sample = meter.read();
                    let Some(watts) = sample.get::<RailPower>() else {
                        continue;
                    };
                    let watts = watts.value() as f64;
                    let mut accumulator = accumulator.lock().unwrap();
                    accumulator.energy_joules += watts * seconds;
                    accumulator.elapsed_seconds += seconds;
                    accumulator.samples += 1;
                }
            }));
        }

        pub(super) fn stop(&mut self) -> Option<PowerReading> {
            self.running.store(false, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
            let accumulator = self.accumulator.lock().unwrap();
            if accumulator.samples == 0 {
                return None;
            }
            let total = if accumulator.elapsed_seconds > 0.0 {
                (accumulator.energy_joules / accumulator.elapsed_seconds) as f32
            } else {
                0.0
            };
            Some(PowerReading {
                cpu: None,
                gpu: None,
                ane: None,
                ram: None,
                total: Watts(total),
                energy: Joules(accumulator.energy_joules as f32),
                samples: accumulator.samples,
            })
        }
    }
}
