use std::{fs::File, io};

/// Positional `read_exact` built on `rustix::io::pread`, so the same code
/// path works on Unix and WASI without pulling in `std::os::*::fs::FileExt`.
pub fn read_exact_at(
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
