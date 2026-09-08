/// Energy and DRAM counters accumulated over one measurement window.
pub struct Reading {
    pub cpu_j: f64,
    pub gpu_j: f64,
    pub ane_j: f64,
    pub ram_j: f64,
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
    pub dram_read_gbps: f64,
    pub dram_write_gbps: f64,
}

impl Reading {
    pub fn energy_total_j(&self) -> f64 {
        self.cpu_j + self.gpu_j + self.ane_j + self.ram_j
    }
}
