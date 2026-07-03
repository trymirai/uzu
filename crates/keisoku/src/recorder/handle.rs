use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use super::{Config, Session};
use crate::{Collector, snapshot::Snapshot, units::Milliseconds};

pub struct RecorderHandle {
    interval: Duration,
    stop_flag: Arc<AtomicBool>,
    snapshots: Arc<Mutex<Vec<Snapshot>>>,
    // Option so both stop() and Drop can take ownership to join.
    worker: Option<std::thread::JoinHandle<()>>,
}

impl RecorderHandle {
    pub fn stop(mut self) -> Session {
        self.stop_flag.store(true, Ordering::Relaxed);
        // Join so the in-flight sample is committed before we read the buffer, and so
        // workers can't pile up across back-to-back recorders. Waits up to one interval.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let snapshots = self.snapshots.lock().map(|mut collected| std::mem::take(&mut *collected)).unwrap_or_default();
        Session {
            interval: Milliseconds(self.interval.as_millis() as u64),
            snapshots,
        }
    }
}

impl Drop for RecorderHandle {
    fn drop(&mut self) {
        // Reached only when stop() was not called, e.g. an error path.
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn start(config: Config) -> RecorderHandle {
    let interval = config.interval;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let snapshots = Arc::new(Mutex::new(Vec::new()));

    let worker = {
        let stop_flag = Arc::clone(&stop_flag);
        let snapshots = Arc::clone(&snapshots);
        std::thread::spawn(move || {
            let mut collector = Collector::new();
            while !stop_flag.load(Ordering::Relaxed) {
                let snapshot = collector.sample(interval);
                if let Ok(mut collected) = snapshots.lock() {
                    collected.push(snapshot);
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
