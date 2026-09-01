use crate::{
    error::KeisokuError,
    units::{Joules, Watts},
};

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

    pub fn start(&mut self) -> Result<(), KeisokuError> {
        self.inner.start()
    }

    pub fn split(&mut self) -> Result<PowerReading, KeisokuError> {
        self.inner.split()
    }

    pub fn stop(&mut self) -> Result<PowerReading, KeisokuError> {
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
    use super::{KeisokuError, PowerReading};
    use crate::{
        Device, Select,
        marker::{Ane, Cpu, EnergyRail, Gpu, Ram},
        units::{Joules, Watts},
    };

    type Rails = Select![EnergyRail<Cpu>, EnergyRail<Gpu>, EnergyRail<Ane>, EnergyRail<Ram>];

    pub struct Inner {
        handle: Option<crate::device::IntervalHandle<Rails>>,
    }

    impl Inner {
        pub fn new() -> Self {
            Self {
                handle: None,
            }
        }

        pub fn start(&mut self) -> Result<(), KeisokuError> {
            let mut handle = Device::interval_measurement::<Rails>();
            handle.start();
            self.handle = Some(handle);
            Ok(())
        }

        pub fn stop(&mut self) -> Result<PowerReading, KeisokuError> {
            let Some(mut handle) = self.handle.take() else {
                return Err(KeisokuError::PowerMeterNotStarted);
            };
            let elapsed = handle.elapsed().as_secs_f64().max(0.001);
            let Some(sample) = handle.stop() else {
                return Err(KeisokuError::PowerReadingUnavailable);
            };
            let cpu_j = sample.get::<EnergyRail<Cpu>>().value() as f64;
            let gpu_j = sample.get::<EnergyRail<Gpu>>().value() as f64;
            let ane_j = sample.get::<EnergyRail<Ane>>().value() as f64;
            let ram_j = sample.get::<EnergyRail<Ram>>().value() as f64;
            let total_j = cpu_j + gpu_j + ane_j + ram_j;
            let to_watts = |joules: f64| Watts((joules / elapsed) as f32);
            Ok(PowerReading {
                cpu: Some(to_watts(cpu_j)),
                gpu: Some(to_watts(gpu_j)),
                ane: Some(to_watts(ane_j)),
                ram: Some(to_watts(ram_j)),
                total: to_watts(total_j),
                energy: Joules(total_j as f32),
                samples: 1,
            })
        }

        pub fn split(&mut self) -> Result<PowerReading, KeisokuError> {
            let reading = self.stop();
            self.start()?;
            reading
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod inner {
    use std::{
        sync::{Arc, Mutex, mpsc},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use super::{KeisokuError, PowerReading};
    use crate::{
        Device,
        units::{Joules, Watts},
    };

    const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

    #[derive(Default)]
    struct Accumulator {
        energy_joules: f64,
        elapsed_seconds: f64,
        samples: u64,
    }

    struct Sampler {
        stop: mpsc::Sender<()>,
        worker: JoinHandle<Result<(), KeisokuError>>,
    }

    pub struct Inner {
        accumulator: Arc<Mutex<Accumulator>>,
        sampler: Option<Sampler>,
    }

    impl Inner {
        pub fn new() -> Self {
            Self {
                accumulator: Arc::new(Mutex::new(Accumulator::default())),
                sampler: None,
            }
        }

        pub fn start(&mut self) -> Result<(), KeisokuError> {
            if self.sampler.is_some() {
                self.stop_sampler()?;
            }
            *self.accumulator.lock()? = Accumulator::default();
            let accumulator = self.accumulator.clone();
            let (stop, stop_receiver) = mpsc::channel();
            let worker = thread::spawn(move || -> Result<(), KeisokuError> {
                let mut device = Device::new();
                let mut last = Instant::now();
                loop {
                    match stop_receiver.recv_timeout(SAMPLE_INTERVAL) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {},
                    }

                    let now = Instant::now();
                    let seconds = now.duration_since(last).as_secs_f64();
                    last = now;
                    let Some(watts) = device.rail_power() else {
                        continue;
                    };
                    let watts = watts.value() as f64;
                    let mut accumulator = accumulator.lock()?;
                    accumulator.energy_joules += watts * seconds;
                    accumulator.elapsed_seconds += seconds;
                    accumulator.samples += 1;
                }
                Ok(())
            });
            self.sampler = Some(Sampler {
                stop,
                worker,
            });
            Ok(())
        }

        pub fn split(&mut self) -> Result<PowerReading, KeisokuError> {
            if self.sampler.is_none() {
                return Err(KeisokuError::PowerMeterNotStarted);
            }
            let accumulator = {
                let mut accumulator = self.accumulator.lock()?;
                std::mem::take(&mut *accumulator)
            };
            Self::reading(accumulator)
        }

        pub fn stop(&mut self) -> Result<PowerReading, KeisokuError> {
            if self.sampler.is_none() {
                return Err(KeisokuError::PowerMeterNotStarted);
            }
            self.stop_sampler()?;
            let accumulator = {
                let mut accumulator = self.accumulator.lock()?;
                std::mem::take(&mut *accumulator)
            };
            Self::reading(accumulator)
        }

        fn stop_sampler(&mut self) -> Result<(), KeisokuError> {
            let Some(sampler) = self.sampler.take() else {
                return Ok(());
            };
            drop(sampler.stop);
            sampler.worker.join().map_err(|_| KeisokuError::SamplingTaskPanicked)?
        }

        fn reading(accumulator: Accumulator) -> Result<PowerReading, KeisokuError> {
            if accumulator.samples == 0 {
                return Err(KeisokuError::PowerReadingUnavailable);
            }
            let total = if accumulator.elapsed_seconds > 0.0 {
                (accumulator.energy_joules / accumulator.elapsed_seconds) as f32
            } else {
                0.0
            };
            Ok(PowerReading {
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
