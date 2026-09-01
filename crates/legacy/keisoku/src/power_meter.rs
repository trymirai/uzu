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
        sync::mpsc,
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use super::{KeisokuError, PowerReading};
    use crate::{Device, units::Joules};

    const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

    #[derive(Default)]
    struct Accumulator {
        joules: Option<f64>,
    }

    enum Command {
        Split {
            boundary: Instant,
            response: mpsc::Sender<Result<PowerReading, KeisokuError>>,
        },
        Stop {
            boundary: Instant,
            response: mpsc::Sender<Result<PowerReading, KeisokuError>>,
        },
    }

    struct Sampler {
        commands: mpsc::Sender<Command>,
        worker: JoinHandle<()>,
    }

    pub struct Inner {
        sampler: Option<Sampler>,
    }

    impl Inner {
        pub fn new() -> Self {
            Self {
                sampler: None,
            }
        }

        pub fn start(&mut self) -> Result<(), KeisokuError> {
            if let Some(sampler) = self.sampler.take() {
                sampler.shutdown()?;
            }
            let (commands, receiver) = mpsc::channel();
            let worker = thread::spawn(move || run(receiver));
            self.sampler = Some(Sampler {
                commands,
                worker,
            });
            Ok(())
        }

        pub fn split(&mut self) -> Result<PowerReading, KeisokuError> {
            self.sampler.as_ref().ok_or(KeisokuError::PowerMeterNotStarted)?.split()
        }

        pub fn stop(&mut self) -> Result<PowerReading, KeisokuError> {
            self.sampler.take().ok_or(KeisokuError::PowerMeterNotStarted)?.stop()
        }
    }

    impl Sampler {
        fn split(&self) -> Result<PowerReading, KeisokuError> {
            let (response, receiver) = mpsc::channel();
            self.commands
                .send(Command::Split {
                    boundary: Instant::now(),
                    response,
                })
                .map_err(|_| KeisokuError::SamplingTaskDisconnected)?;
            receiver.recv().map_err(|_| KeisokuError::SamplingTaskDisconnected)?
        }

        fn stop(self) -> Result<PowerReading, KeisokuError> {
            let (response, receiver) = mpsc::channel();
            let reading = self
                .commands
                .send(Command::Stop {
                    boundary: Instant::now(),
                    response,
                })
                .map_err(|_| KeisokuError::SamplingTaskDisconnected)
                .and_then(|()| receiver.recv().map_err(|_| KeisokuError::SamplingTaskDisconnected))
                .and_then(|reading| reading);
            self.worker.join().map_err(|_| KeisokuError::SamplingTaskPanicked)?;
            reading
        }

        fn shutdown(self) -> Result<(), KeisokuError> {
            drop(self.commands);
            self.worker.join().map_err(|_| KeisokuError::SamplingTaskPanicked)
        }
    }

    fn run(commands: mpsc::Receiver<Command>) {
        let mut device = Device::new();
        let mut accumulator = Accumulator::default();
        let mut last_sample = Instant::now();

        loop {
            match commands.recv_timeout(SAMPLE_INTERVAL) {
                Ok(Command::Split {
                    boundary,
                    response,
                }) => {
                    sample(&mut device, &mut accumulator, &mut last_sample, boundary);
                    let _ = response.send(reading(std::mem::take(&mut accumulator)));
                },
                Ok(Command::Stop {
                    boundary,
                    response,
                }) => {
                    sample(&mut device, &mut accumulator, &mut last_sample, boundary);
                    let _ = response.send(reading(accumulator));
                    break;
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    sample(&mut device, &mut accumulator, &mut last_sample, Instant::now());
                },
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    fn sample(
        device: &mut Device,
        accumulator: &mut Accumulator,
        last_sample: &mut Instant,
        boundary: Instant,
    ) {
        let elapsed = boundary.saturating_duration_since(*last_sample);
        *last_sample = boundary;
        if let Some(energy) = device.rail_energy(elapsed) {
            *accumulator.joules.get_or_insert(0.0) += f64::from(energy.value());
        }
    }

    fn reading(accumulator: Accumulator) -> Result<PowerReading, KeisokuError> {
        accumulator
            .joules
            .map(|joules| PowerReading::Total {
                total: Joules(joules as f32),
            })
            .ok_or(KeisokuError::PowerReadingUnavailable)
    }
}
