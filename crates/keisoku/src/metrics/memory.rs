use serde::{Deserialize, Serialize};

use crate::units::Bytes;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub ram_total: Bytes,
    pub ram_usage: Bytes,
    pub swap_total: Bytes,
    pub swap_usage: Bytes,
}

impl MemoryMetrics {
    #[cfg(target_os = "macos")]
    pub(crate) fn read() -> Option<MemoryMetrics> {
        let mut ram_total = 0u64;
        unsafe {
            let mut name = [libc::CTL_HW, libc::HW_MEMSIZE];
            let mut size = core::mem::size_of::<u64>();
            if libc::sysctl(
                name.as_mut_ptr(),
                name.len() as _,
                &mut ram_total as *mut _ as *mut _,
                &mut size,
                core::ptr::null_mut(),
                0,
            ) != 0
            {
                return None;
            }
        }

        let ram_usage = unsafe {
            let mut count: u32 = libc::HOST_VM_INFO64_COUNT as _;
            let mut statistics = core::mem::zeroed::<libc::vm_statistics64>();
            #[allow(deprecated)]
            let result = libc::host_statistics64(
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                &mut statistics as *mut _ as *mut _,
                &mut count,
            );
            if result != 0 {
                return None;
            }
            let page_size = libc::sysconf(libc::_SC_PAGESIZE) as u64;
            (statistics.active_count as u64
                + statistics.inactive_count as u64
                + statistics.wire_count as u64
                + statistics.speculative_count as u64
                + statistics.compressor_page_count as u64
                - statistics.purgeable_count as u64
                - statistics.external_page_count as u64)
                * page_size
        };

        let (swap_usage, swap_total) = unsafe {
            let mut name = [libc::CTL_VM, libc::VM_SWAPUSAGE];
            let mut size = core::mem::size_of::<libc::xsw_usage>();
            let mut swap = core::mem::zeroed::<libc::xsw_usage>();
            if libc::sysctl(
                name.as_mut_ptr(),
                name.len() as _,
                &mut swap as *mut _ as *mut _,
                &mut size,
                core::ptr::null_mut(),
                0,
            ) == 0
            {
                (swap.xsu_used, swap.xsu_total)
            } else {
                (0, 0)
            }
        };

        Some(MemoryMetrics {
            ram_total: Bytes(ram_total),
            ram_usage: Bytes(ram_usage),
            swap_total: Bytes(swap_total),
            swap_usage: Bytes(swap_usage),
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub(crate) fn read() -> Option<MemoryMetrics> {
        use sysinfo::System;
        let mut system = System::new();
        system.refresh_memory();
        Some(MemoryMetrics {
            ram_total: Bytes(system.total_memory()),
            ram_usage: Bytes(system.used_memory()),
            swap_total: Bytes(system.total_swap()),
            swap_usage: Bytes(system.used_swap()),
        })
    }
}
