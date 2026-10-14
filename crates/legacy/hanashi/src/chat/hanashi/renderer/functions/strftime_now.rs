#[cfg(not(target_arch = "wasm32"))]
use std::{ffi::CString, mem::MaybeUninit};

/// Formats the current local time according to the provided `strftime` pattern
#[cfg(not(target_arch = "wasm32"))]
pub fn strftime_now(format_string: String) -> String {
    let Ok(c_format) = CString::new(format_string) else {
        return String::new();
    };

    unsafe {
        let now = libc::time(std::ptr::null_mut());
        if now == -1 {
            return String::new();
        }

        let mut tm = MaybeUninit::<libc::tm>::uninit();
        if libc::localtime_r(&now, tm.as_mut_ptr()).is_null() {
            return String::new();
        }

        let tm = tm.assume_init();
        let mut buf_len = 128_usize;

        loop {
            let mut buffer = vec![0u8; buf_len];
            let written =
                libc::strftime(buffer.as_mut_ptr() as *mut libc::c_char, buffer.len(), c_format.as_ptr(), &tm);

            if written > 0 {
                buffer.truncate(written as usize);
                return String::from_utf8_lossy(&buffer).into_owned();
            }

            if buf_len >= 4096 {
                return String::new();
            }

            buf_len *= 2;
        }
    }
}

// Browser WASM has no libc clock or localtime_r. Use the device's current
// offset so templates retain the same local-time semantics as desktop.
#[cfg(target_arch = "wasm32")]
pub fn strftime_now(format_string: String) -> String {
    use std::fmt::Write;
    let now = js_sys::Date::new_0();
    let Some(utc) = chrono::DateTime::from_timestamp_millis(now.get_time() as i64) else {
        return String::new();
    };
    let Some(offset) = chrono::FixedOffset::west_opt((now.get_timezone_offset() * 60.0) as i32) else {
        return String::new();
    };
    let mut rendered = String::new();
    if write!(&mut rendered, "{}", utc.with_timezone(&offset).format(&format_string)).is_err() {
        return String::new();
    }
    rendered
}
