#[derive(Debug, Default, Clone)]
pub struct SocSample {
    pub cpu_power: f32,
    pub gpu_power: f32,
    pub ane_power: f32,
    pub ram_power: f32,
    pub gpu_ram_power: f32,
    pub total_power: f32,
    pub ecpu_usage: (u32, f32),
    pub pcpu_usage: (u32, f32),
    pub cpu_usage_percent: f32,
    pub gpu_usage: (u32, f32),
    pub ane_active_percent: f32,
    pub ane_read_gbps: f32,
    pub ane_write_gbps: f32,
    pub dram_read_gbps: f32,
    pub dram_write_gbps: f32,
}
