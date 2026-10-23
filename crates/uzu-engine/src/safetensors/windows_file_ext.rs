use std::{fs::File, io, os::windows::fs::FileExt as _};

/// The Unix `FileExt::read_exact_at` the loader reads with, over Windows positioned reads, which may return short.
pub trait WindowsFileExt {
    fn read_exact_at(
        &self,
        buffer: &mut [u8],
        offset: u64,
    ) -> io::Result<()>;
}

impl WindowsFileExt for File {
    fn read_exact_at(
        &self,
        mut buffer: &mut [u8],
        mut offset: u64,
    ) -> io::Result<()> {
        while !buffer.is_empty() {
            match self.seek_read(buffer, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(read) => {
                    buffer = &mut buffer[read..];
                    offset += read as u64;
                },
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {},
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}
