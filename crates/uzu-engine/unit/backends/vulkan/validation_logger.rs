use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::backends::vulkan::VkLogger;

/// Counts validation-layer errors so a test fails on them instead of only logging them.
#[derive(Clone, Default)]
pub struct ValidationLogger {
    errors: Arc<AtomicUsize>,
}

impl ValidationLogger {
    pub fn assert_clean(&self) {
        assert_eq!(self.errors.load(Ordering::SeqCst), 0, "Vulkan validation reported errors");
    }
}

impl VkLogger for ValidationLogger {
    fn v(
        &self,
        _msg: &str,
    ) {
    }

    fn i(
        &self,
        _msg: &str,
    ) {
    }

    fn d(
        &self,
        _msg: &str,
    ) {
    }

    fn w(
        &self,
        msg: &str,
    ) {
        eprintln!("[Warning]: {msg}");
    }

    fn e(
        &self,
        msg: &str,
    ) {
        eprintln!("[Error]: {msg}");
        self.errors.fetch_add(1, Ordering::SeqCst);
    }
}
