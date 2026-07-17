use std::{fs::File, io};

/// Positional `read_exact` built on `rustix::io::pread`, so the same code
/// path works on Unix and WASI without pulling in `std::os::*::fs::FileExt`.
pub fn file_read_exact_at(
    file: &File,
    mut buf: &mut [u8],
    mut offset: u64,
) -> io::Result<()> {
    while !buf.is_empty() {
        match rustix::io::pread(file, &mut *buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            },
            Err(rustix::io::Errno::INTR) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

pub fn disable_page_cache(file: &File) -> io::Result<()> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use std::os::fd::AsRawFd;

        const F_NOCACHE: libc::c_int = 48;
        const F_RDAHEAD: libc::c_int = 45;
        const F_GLOBAL_NOCACHE: libc::c_int = 55;
        let fd = file.as_raw_fd();
        for (command, value) in [(F_GLOBAL_NOCACHE, 1), (F_NOCACHE, 1), (F_RDAHEAD, 0)] {
            if unsafe { libc::fcntl(fd, command, value) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let _ = file;
    Ok(())
}
