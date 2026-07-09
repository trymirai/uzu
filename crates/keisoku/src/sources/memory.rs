use crate::metrics::MemoryMetrics;

#[cfg(target_os = "macos")]
pub(crate) fn read_memory() -> Option<MemoryMetrics> {
    crate::sys::read_memory()
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn read_memory() -> Option<MemoryMetrics> {
    use sysinfo::System;

    use crate::units::Bytes;
    let mut system = System::new();
    system.refresh_memory();
    Some(MemoryMetrics {
        ram_total: Bytes(system.total_memory()),
        ram_usage: Bytes(system.used_memory()),
        swap_total: Bytes(system.total_swap()),
        swap_usage: Bytes(system.used_swap()),
    })
}
