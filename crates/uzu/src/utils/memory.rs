#[cfg(not(target_arch = "wasm32"))]
pub fn get_free_ram() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() - sys.used_memory()
}

#[cfg(target_arch = "wasm32")]
pub fn get_free_ram() -> u64 {
    use core::arch::wasm32::memory_size;
    const TOTAL_AVAILABLE: u64 = 4 * 1024 * 1024 * 1024;
    const PAGE_SIZE: u64 = 65536;

    let used = (memory_size::<0>() as u64) * PAGE_SIZE;
    TOTAL_AVAILABLE - used
}

#[cfg(not(target_arch = "wasm32"))]
pub fn get_free_swap() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() - sys.used_memory()
}

#[cfg(target_arch = "wasm32")]
pub fn get_free_swap() -> u64 {
    0u64
}
