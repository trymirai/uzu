use std::{
    io::Write,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{HardwareError, HardwareSession};

/// Ends the render scope's session even while a worker retains a session reference.
pub struct HardwareSessionGuard {
    closed: Arc<AtomicBool>,
    session: Arc<Mutex<HardwareSession>>,
}

impl HardwareSessionGuard {
    pub fn new() -> Self {
        let closed = Arc::new(AtomicBool::new(false));
        Self {
            session: Arc::new(Mutex::new(HardwareSession::new(Arc::clone(&closed)))),
            closed,
        }
    }

    pub fn session(&self) -> Arc<Mutex<HardwareSession>> {
        Arc::clone(&self.session)
    }

    pub fn restore(&mut self) -> Result<(), HardwareError> {
        self.closed.store(true, Ordering::Release);
        self.session
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::error!("Recovering poisoned hardware controls to restore system settings");
                poisoned.into_inner()
            })
            .restore()
    }
}

impl Drop for HardwareSessionGuard {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            let _ = writeln!(std::io::stderr(), "Unable to restore hardware settings: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::interactive::hardware::PerformanceMode;

    #[test]
    fn owner_drop_closes_before_waiting_for_the_worker_lock() {
        let guard = HardwareSessionGuard::new();
        let closed = Arc::clone(&guard.closed);
        let session = guard.session();
        let (ready, locked) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut session = session.lock().unwrap();
            ready.send(()).unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while !closed.load(Ordering::Acquire) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            assert!(closed.load(Ordering::Acquire));
            assert!(matches!(session.apply(PerformanceMode::Auto), Err(HardwareError::Closed)));
        });
        locked.recv().unwrap();
        drop(guard);
        worker.join().unwrap();
    }

    #[test]
    fn unwind_closes_a_retained_session_without_authorization() {
        let mut retained = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let guard = HardwareSessionGuard::new();
            retained = Some((Arc::clone(&guard.closed), guard.session()));
            panic!("render failed");
        }));
        assert!(result.is_err());
        let (closed, session) = retained.unwrap();
        assert!(closed.load(Ordering::Acquire));
        assert!(matches!(session.lock().unwrap().apply(PerformanceMode::Auto), Err(HardwareError::Closed)));
    }
}
