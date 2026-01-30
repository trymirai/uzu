use std::time::Duration;

#[derive(Clone, Copy, Default)]
pub struct AllocationStats {
    pub total_allocs: usize,
    pub total_frees: usize,
    pub total_alloc_time_ns: u128,
    pub total_free_time_ns: u128,
    pub total_bytes_requested: usize,
}

impl AllocationStats {
    pub fn record_alloc(
        &mut self,
        size: usize,
        duration: Duration,
    ) {
        self.total_allocs += 1;
        self.total_alloc_time_ns += duration.as_nanos();
        self.total_bytes_requested += size;
    }

    pub fn record_free(
        &mut self,
        duration: Duration,
    ) {
        self.total_frees += 1;
        self.total_free_time_ns += duration.as_nanos();
    }

    pub fn avg_alloc_time_us(&self) -> f64 {
        if self.total_allocs == 0 {
            0.0
        } else {
            self.total_alloc_time_ns as f64 / self.total_allocs as f64 / 1000.0
        }
    }

    pub fn avg_free_time_us(&self) -> f64 {
        if self.total_frees == 0 {
            0.0
        } else {
            self.total_free_time_ns as f64 / self.total_frees as f64 / 1000.0
        }
    }
}
