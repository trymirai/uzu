use std::{ffi::CString, mem::MaybeUninit};

/// Formats the current local time according to the provided `strftime` pattern
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
