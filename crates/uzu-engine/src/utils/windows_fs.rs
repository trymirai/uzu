use std::{fs::File, io, os::windows::fs::FileExt as WindowsFileExt};

/// Positional reads with the same shape as `std::os::unix::fs::FileExt`, built on `seek_read`.
pub trait FileExt {
    fn read_exact_at(
        &self,
        buf: &mut [u8],
        offset: u64,
    ) -> io::Result<()>;
}

impl FileExt for File {
    fn read_exact_at(
        &self,
        mut buf: &mut [u8],
        mut offset: u64,
    ) -> io::Result<()> {
        while !buf.is_empty() {
            match self.seek_read(buf, offset) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "failed to fill whole buffer")),
                Ok(read) => {
                    buf = &mut buf[read..];
                    offset += read as u64;
                },
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {},
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}
