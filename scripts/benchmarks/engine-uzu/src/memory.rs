use std::ffi::CStr;

use anyhow::bail;

pub use crate::system::memory_counters_t as MemoryCounters;
use crate::system::{get_memory_counters, memory_counters_error_string};

impl MemoryCounters {
    pub fn collect() -> anyhow::Result<Self> {
        let mut counters = Self::default();
        let result = unsafe { get_memory_counters(&mut counters, false) };
        if result != 0 {
            // SAFETY: memory_counters_error_string returns a static NUL-terminated string.
            let detail = unsafe { CStr::from_ptr(memory_counters_error_string(result)) }.to_string_lossy();
            bail!("get_memory_counters failed ({result}): {detail}");
        }
        Ok(counters)
    }
}
