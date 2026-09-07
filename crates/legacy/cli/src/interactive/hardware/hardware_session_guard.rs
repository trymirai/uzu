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
