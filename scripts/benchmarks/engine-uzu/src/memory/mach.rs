use std::ffi::{CStr, c_char, c_int};

// Keep this layout in sync with memory_counters_t in common-cpp/src/memory_counters.h.
#[repr(C)]
#[derive(Default)]
pub struct MemoryCounters {
    pub pid: i32,
    pub phys_footprint: u64,
    pub resident_size: u64,
    pub resident_size_peak: u64,
    pub device: u64,
    pub device_peak: u64,
    pub internal: u64,
    pub compressed: u64,
    pub graphics_footprint: u64,
    pub graphics_footprint_compressed: u64,
    pub graphics_nofootprint: u64,
    pub graphics_nofootprint_compressed: u64,
    pub graphics_total: u64,
    pub malloc_allocated: u64,
    pub malloc_in_use: u64,
    pub malloc_max_in_use: u64,
}

impl MemoryCounters {
    pub fn collect() -> anyhow::Result<Self> {
        let mut counters = Self::default();
        // SAFETY: counters is writable and has the layout declared by memory_counters_t.
        let result = unsafe { get_memory_counters(&mut counters, false) };
        if result != 0 {
            // SAFETY: the C helper accepts any kern_return_t error code.
            let message = unsafe { memory_counters_error_string(result) };
            let detail = if message.is_null() {
                "Unknown error".into()
            } else {
                // SAFETY: non-null results refer to NUL-terminated C error strings.
                unsafe { CStr::from_ptr(message) }.to_string_lossy()
            };
            anyhow::bail!("get_memory_counters failed ({result}): {detail}");
        }
        Ok(counters)
    }
}

unsafe extern "C" {

    fn get_memory_counters(
        counters: *mut MemoryCounters,
        with_malloc_zone_stats: bool,
    ) -> c_int;

    fn memory_counters_error_string(result: c_int) -> *const c_char;
}
