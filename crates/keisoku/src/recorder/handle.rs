use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use super::{Config, Session};
use crate::{Collector, snapshot::Snapshot, units::Milliseconds};

pub struct RecorderHandle {
    interval: Duration,
    stop_flag: Arc<AtomicBool>,
    snapshots: Arc<Mutex<Vec<Snapshot>>>,
    // Option so both stop() and Drop can take ownership via take().
    worker: Option<std::thread::JoinHandle<()>>,
}

impl RecorderHandle {
    pub fn stop(mut self) -> Session {
        self.stop_flag.store(true, Ordering::Relaxed);
        // Join so the in-flight sample completes and its snapshot is committed before
        // we read the buffer. This also prevents workers from piling up across
        // back-to-back recorder instances (e.g. in benchmark loops).
        // Max wait = one sampling interval (~100 ms).
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        let snapshots = self.snapshots.lock().map(|mut s| std::mem::take(&mut *s)).unwrap_or_default();
        Session {
            interval: Milliseconds(self.interval.as_millis() as u64),
            snapshots,
        }
    }
}

impl Drop for RecorderHandle {
    fn drop(&mut self) {
        // Reached only when stop() was not called (e.g. an error path). Signal the
        // thread and join for a clean exit; max wait = one sampling interval.
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

pub fn start(config: Config) -> RecorderHandle {
    let interval = config.interval;
    let stop_flag = Arc::new(AtomicBool::new(false));
    // Vec<Snapshot> grows at one entry per interval. At 100 ms intervals a 60-second
    // response produces ~600 entries — negligible memory for the per-response use case.
    let snapshots = Arc::new(Mutex::new(Vec::new()));

    let worker = {
        let stop_flag = Arc::clone(&stop_flag);
        let snapshots = Arc::clone(&snapshots);
        std::thread::spawn(move || {
            let mut collector = Collector::new();
            // started_at is set after Collector::new() so elapsed values reflect only
            // actual sampling time, not collector initialisation overhead.
            let started_at = Instant::now();
            while !stop_flag.load(Ordering::Relaxed) {
                let mut snapshot = collector.sample(interval);
                snapshot.elapsed = Milliseconds(started_at.elapsed().as_millis() as u64);
                if let Ok(mut guard) = snapshots.lock() {
                    guard.push(snapshot);
                }
            }
        })
    };

    RecorderHandle {
        interval,
        stop_flag,
        snapshots,
        worker: Some(worker),
    }
}
