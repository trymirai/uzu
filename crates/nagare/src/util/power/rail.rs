#![cfg(all(target_vendor = "apple", not(target_os = "macos")))]

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use keisoku::{Instant as InstantMeter, RailPower, Select};
use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Default)]
struct Accumulator {
    energy_joules: f64,
    elapsed_seconds: f64,
    samples: i64,
}

pub struct RailPowerRecorder {
    running: Arc<AtomicBool>,
    accumulator: Arc<Mutex<Accumulator>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl RailPowerRecorder {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            accumulator: Arc::new(Mutex::new(Accumulator::default())),
            worker: Mutex::new(None),
        }
    }
}

impl PowerRecorder for RailPowerRecorder {
    fn begin(&self) {
        *self.accumulator.lock().unwrap() = Accumulator::default();
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let accumulator = self.accumulator.clone();
        let worker = thread::spawn(move || {
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
        });
        *self.worker.lock().unwrap() = Some(worker);
    }

    fn finish(&self) -> Option<ChatReplyPowerStats> {
        self.running.store(false, Ordering::SeqCst);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
        let accumulator = self.accumulator.lock().unwrap();
        if accumulator.samples == 0 {
            return None;
        }
        let average_total_watts = if accumulator.elapsed_seconds > 0.0 {
            accumulator.energy_joules / accumulator.elapsed_seconds
        } else {
            0.0
        };
        Some(ChatReplyPowerStats {
            samples_count: accumulator.samples,
            average_cpu_watts: 0.0,
            average_gpu_watts: 0.0,
            average_ane_watts: 0.0,
            average_ram_watts: 0.0,
            average_total_watts,
            energy_joules: accumulator.energy_joules,
        })
    }
}
