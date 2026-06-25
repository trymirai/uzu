use serde::{Deserialize, Serialize};

use crate::{Collector, units::Bytes};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Device {
    pub os: String,
    pub chip: String,
    pub ram_total: Bytes,
    pub efficiency_cores: u8,
    pub performance_cores: u8,
    pub gpu_cores: u8,
}

impl Device {
    pub(crate) fn detect(#[allow(unused_variables)] collector: &Collector) -> Device {
        let os = sysinfo::System::long_os_version().unwrap_or_default();
        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        let chip = system.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
        let ram_total = Bytes(system.total_memory());

        #[cfg(target_os = "macos")]
        if let Some(soc) = collector.soc() {
            return Device {
                os,
                chip: if soc.chip_name.is_empty() {
                    chip
                } else {
                    soc.chip_name.clone()
                },
                ram_total,
                efficiency_cores: soc.ecpu_cores,
                performance_cores: soc.pcpu_cores,
                gpu_cores: soc.gpu_cores,
            };
        }

        #[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
        #[allow(clippy::needless_return)]
        {
            let (performance_cores, efficiency_cores) = perflevel_cores();
            return Device {
                os,
                chip: sysctl_string("hw.machine").filter(|model| !model.is_empty()).unwrap_or(chip),
                ram_total,
                efficiency_cores,
                performance_cores,
                gpu_cores: 0,
            };
        }

        #[cfg(not(all(target_vendor = "apple", not(target_os = "macos"))))]
        Device {
            os,
            chip,
            ram_total,
            ..Default::default()
        }
    }
}

#[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
fn sysctl_string(name: &str) -> Option<String> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut len = 0usize;
    let probe = unsafe { libc::sysctlbyname(name.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) };
    if probe != 0 || len == 0 {
        return None;
    }
    let mut buffer = vec![0u8; len];
    let read =
        unsafe { libc::sysctlbyname(name.as_ptr(), buffer.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) };
    if read != 0 {
        return None;
    }
    if let Some(nul) = buffer.iter().position(|&byte| byte == 0) {
        buffer.truncate(nul);
    }
    String::from_utf8(buffer).ok()
}

#[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
fn sysctl_u32(name: &str) -> Option<u32> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = 0u32;
    let mut len = std::mem::size_of::<u32>();
    let read = unsafe {
        libc::sysctlbyname(name.as_ptr(), std::ptr::addr_of_mut!(value).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    (read == 0).then_some(value)
}

#[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
fn perflevel_cores() -> (u8, u8) {
    let performance = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);
    let efficiency = if sysctl_u32("hw.nperflevels").unwrap_or(1) > 1 {
        sysctl_u32("hw.perflevel1.logicalcpu").unwrap_or(0)
    } else {
        0
    };
    (performance as u8, efficiency as u8)
}
