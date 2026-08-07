use crate::units::{Joules, Watts};

pub struct PowerReading {
    pub cpu: Option<Watts>,
    pub gpu: Option<Watts>,
    pub ane: Option<Watts>,
    pub ram: Option<Watts>,
    pub total: Watts,
    pub energy: Joules,
    pub samples: u64,
    pub dram_read_bytes: Option<u64>,
    pub dram_write_bytes: Option<u64>,
    pub dram_read_gbps: Option<f32>,
    pub dram_write_gbps: Option<f32>,
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
        Device, Select,
        marker::{Ane, Cpu, DramBytes, DramHistogram, DramRead, DramWrite, EnergyRail, Gpu, Ram},
        units::{Joules, Watts},
    };

    type Rails = Select![
        EnergyRail<Cpu>,
        EnergyRail<Gpu>,
        EnergyRail<Ane>,
        EnergyRail<Ram>,
        DramBytes<DramRead>,
        DramBytes<DramWrite>,
        DramHistogram<DramRead>,
        DramHistogram<DramWrite>,
    ];

    pub(super) struct Inner {
        handle: Option<crate::device::IntervalHandle<Rails>>,
    }

    impl Inner {
        pub(super) fn new() -> Self {
            Self {
                handle: None,
            }
        }

        pub(super) fn start(&mut self) {
            let mut handle = Device::interval_measurement::<Rails>();
            handle.start();
            self.handle = Some(handle);
        }

        pub(super) fn stop(&mut self) -> Option<PowerReading> {
            let mut handle = self.handle.take()?;
            let elapsed = handle.elapsed().as_secs_f64().max(0.001);
            let sample = handle.stop()?;
            let cpu_j = sample.get::<EnergyRail<Cpu>>().value() as f64;
            let gpu_j = sample.get::<EnergyRail<Gpu>>().value() as f64;
            let ane_j = sample.get::<EnergyRail<Ane>>().value() as f64;
            let ram_j = sample.get::<EnergyRail<Ram>>().value() as f64;
            let total_j = cpu_j + gpu_j + ane_j + ram_j;
            let to_watts = |joules: f64| Watts((joules / elapsed) as f32);
            Some(PowerReading {
                cpu: Some(to_watts(cpu_j)),
                gpu: Some(to_watts(gpu_j)),
                ane: Some(to_watts(ane_j)),
                ram: Some(to_watts(ram_j)),
                total: to_watts(total_j),
                energy: Joules(total_j as f32),
                samples: 1,
                dram_read_bytes: Some(sample.get::<DramBytes<DramRead>>().value()),
                dram_write_bytes: Some(sample.get::<DramBytes<DramWrite>>().value()),
                dram_read_gbps: Some(sample.get::<DramHistogram<DramRead>>().value()),
                dram_write_gbps: Some(sample.get::<DramHistogram<DramWrite>>().value()),
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
            *self.accumulator.lock().expect("power meter accumulator poisoned") = Accumulator::default();
            self.running.store(true, Ordering::Relaxed);
            let running = self.running.clone();
            let accumulator = self.accumulator.clone();
            self.worker = Some(thread::spawn(move || {
                let mut device = Device::new();
                let mut last = Instant::now();
                while running.load(Ordering::Relaxed) {
                    thread::sleep(SAMPLE_INTERVAL);
                    let now = Instant::now();
                    let seconds = now.duration_since(last).as_secs_f64();
                    last = now;
                    let Some(watts) = device.rail_power() else {
                        continue;
                    };
                    let watts = watts.value() as f64;
                    let mut accumulator = accumulator.lock().expect("power meter accumulator poisoned");
                    accumulator.energy_joules += watts * seconds;
                    accumulator.elapsed_seconds += seconds;
                    accumulator.samples += 1;
                }
            }));
        }

        pub(super) fn stop(&mut self) -> Option<PowerReading> {
            self.running.store(false, Ordering::Relaxed);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
            let accumulator = self.accumulator.lock().expect("power meter accumulator poisoned");
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
                dram_read_bytes: None,
                dram_write_bytes: None,
                dram_read_gbps: None,
                dram_write_gbps: None,
            })
        }
    }
}
