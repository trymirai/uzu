use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use super::{Config, Device, Marker, Session};
use crate::{Collector, snapshot::Snapshot, units::Milliseconds};

/// Handle to a running recorder. Drop or call [`stop`](RecorderHandle::stop).
pub struct RecorderHandle {
    started_at: Instant,
    interval: Duration,
    stop_flag: Arc<AtomicBool>,
    markers: Arc<Mutex<Vec<Marker>>>,
    worker: Option<JoinHandle<(Device, Vec<Snapshot>)>>,
}

impl RecorderHandle {
    /// Annotates the timeline at the current elapsed time.
    pub fn mark(
        &self,
        label: impl Into<String>,
    ) {
        let elapsed = Milliseconds(self.started_at.elapsed().as_millis() as u64);
        if let Ok(mut markers) = self.markers.lock() {
            markers.push(Marker {
                elapsed,
                label: label.into(),
            });
        }
    }

    /// Stops sampling and returns the recorded session.
    pub fn stop(mut self) -> Session {
        self.stop_flag.store(true, Ordering::Relaxed);
        let (device, snapshots) = match self.worker.take().map(JoinHandle::join) {
            Some(Ok(result)) => result,
            _ => (Device::default(), Vec::new()),
        };
        let markers = self.markers.lock().map(|markers| markers.clone()).unwrap_or_default();
        Session {
            device,
            interval: Milliseconds(self.interval.as_millis() as u64),
            snapshots,
            markers,
        }
    }
}

impl Drop for RecorderHandle {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Starts a background recorder sampling at `config.interval`.
pub fn start(config: Config) -> RecorderHandle {
    let started_at = Instant::now();
    let interval = config.interval;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let markers = Arc::new(Mutex::new(Vec::new()));

    let worker = {
        let stop_flag = Arc::clone(&stop_flag);
        // The Collector holds raw IOReport pointers (`!Send`), so it's created
        // and owned entirely inside the worker thread.
        std::thread::spawn(move || {
            let mut collector = Collector::new();
            let device = Device::detect(&collector);
            let mut snapshots = Vec::new();
            while !stop_flag.load(Ordering::Relaxed) {
                let mut snapshot = collector.sample(interval);
                snapshot.elapsed = Milliseconds(started_at.elapsed().as_millis() as u64);
                snapshots.push(snapshot);
            }
            (device, snapshots)
        })
    };

    RecorderHandle {
        started_at,
        interval,
        stop_flag,
        markers,
        worker: Some(worker),
    }
}
