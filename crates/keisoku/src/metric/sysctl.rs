pub(super) fn sysctl_string(name: &str) -> Option<String> {
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

fn sysctl_u32(name: &str) -> Option<u32> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = 0u32;
    let mut len = std::mem::size_of::<u32>();
    let read = unsafe {
        libc::sysctlbyname(name.as_ptr(), std::ptr::addr_of_mut!(value).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    (read == 0).then_some(value)
}

pub(super) fn perflevel_cores() -> (u8, u8) {
    let performance = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);
    let efficiency = if sysctl_u32("hw.nperflevels").unwrap_or(1) > 1 {
        sysctl_u32("hw.perflevel1.logicalcpu").unwrap_or(0)
    } else {
        0
    };
    (performance as u8, efficiency as u8)
}
