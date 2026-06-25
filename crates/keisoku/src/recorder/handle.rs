use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use super::{Config, Device, Marker, Session};
use crate::{Collector, snapshot::Snapshot, units::Milliseconds};

#[derive(Default)]
struct Recording {
    device: Device,
    snapshots: Vec<Snapshot>,
}

pub struct RecorderHandle {
    started_at: Instant,
    interval: Duration,
    stop_flag: Arc<AtomicBool>,
    markers: Arc<Mutex<Vec<Marker>>>,
    recording: Arc<Mutex<Recording>>,
}

impl RecorderHandle {
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

    pub fn stop(self) -> Session {
        // Signal the sampler and return what's collected so far without joining, so the
        // caller never waits for the in-flight ~interval sample (which would also span
        // post-work idle time). The detached worker exits on its next flag check.
        self.stop_flag.store(true, Ordering::Relaxed);
        let (device, snapshots) = self
            .recording
            .lock()
            .map(|recording| (recording.device.clone(), recording.snapshots.clone()))
            .unwrap_or_default();
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
    }
}

pub fn start(config: Config) -> RecorderHandle {
    let started_at = Instant::now();
    let interval = config.interval;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let markers = Arc::new(Mutex::new(Vec::new()));
    let recording = Arc::new(Mutex::new(Recording::default()));

    {
        let stop_flag = Arc::clone(&stop_flag);
        let recording = Arc::clone(&recording);
        std::thread::spawn(move || {
            let mut collector = Collector::new();
            let device = Device::detect(&collector);
            if let Ok(mut recording) = recording.lock() {
                recording.device = device;
            }
            while !stop_flag.load(Ordering::Relaxed) {
                let mut snapshot = collector.sample(interval);
                snapshot.elapsed = Milliseconds(started_at.elapsed().as_millis() as u64);
                if let Ok(mut recording) = recording.lock() {
                    recording.snapshots.push(snapshot);
                }
            }
        });
    }

    RecorderHandle {
        started_at,
        interval,
        stop_flag,
        markers,
        recording,
    }
}
