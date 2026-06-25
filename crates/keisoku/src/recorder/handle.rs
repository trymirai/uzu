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
}

impl RecorderHandle {
    pub fn stop(self) -> Session {
        // Signal the sampler and return what's collected so far without joining, so the
        // caller never waits for the in-flight ~interval sample (which would also span
        // post-work idle time). The detached worker exits on its next flag check.
        self.stop_flag.store(true, Ordering::Relaxed);
        // Move the buffer out rather than cloning; the detached worker won't read it.
        let snapshots = self.snapshots.lock().map(|mut snapshots| std::mem::take(&mut *snapshots)).unwrap_or_default();
        Session {
            interval: Milliseconds(self.interval.as_millis() as u64),
            snapshots,
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
    let snapshots = Arc::new(Mutex::new(Vec::new()));

    {
        let stop_flag = Arc::clone(&stop_flag);
        let snapshots = Arc::clone(&snapshots);
        std::thread::spawn(move || {
            let mut collector = Collector::new();
            while !stop_flag.load(Ordering::Relaxed) {
                let mut snapshot = collector.sample(interval);
                snapshot.elapsed = Milliseconds(started_at.elapsed().as_millis() as u64);
                if let Ok(mut snapshots) = snapshots.lock() {
                    snapshots.push(snapshot);
                }
            }
        });
    }

    RecorderHandle {
        interval,
        stop_flag,
        snapshots,
    }
}
