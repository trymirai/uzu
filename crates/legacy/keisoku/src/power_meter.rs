use crate::{error::KeisokuError, units::Joules};

pub enum PowerReading {
    Total {
        total: Joules,
    },
    Components {
        cpu: Joules,
        gpu: Joules,
        ane: Joules,
        ram: Joules,
    },
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
            let Some(sample) = handle.stop() else {
                return Err(KeisokuError::PowerReadingUnavailable);
            };
            Ok(PowerReading::Components {
                cpu: *sample.get::<EnergyRail<Cpu>>(),
                gpu: *sample.get::<EnergyRail<Gpu>>(),
                ane: *sample.get::<EnergyRail<Ane>>(),
                ram: *sample.get::<EnergyRail<Ram>>(),
            })
        }

        pub fn split(&mut self) -> Result<PowerReading, KeisokuError> {
            let reading = self.stop();
            self.start()?;
            reading
        }
    }
}

#[cfg(target_os = "ios")]
mod inner {
    use std::{
        sync::{Arc, Mutex, mpsc},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use super::{KeisokuError, PowerReading};
    use crate::{Device, units::Joules};

    const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

    #[derive(Default)]
    struct Accumulator {
        energy_joules: f64,
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
                    let elapsed = now.duration_since(last);
                    last = now;
                    let Some(energy) = device.rail_energy(elapsed) else {
                        continue;
                    };
                    let mut accumulator = accumulator.lock()?;
                    accumulator.energy_joules += energy.value() as f64;
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
            Ok(PowerReading::Total {
                total: Joules(accumulator.energy_joules as f32),
            })
        }
    }
}
