use std::{io, path::Path};

pub use async_fs::read;
use futures::{
    io::{AsyncReadExt, AsyncSeekExt, SeekFrom},
    lock::Mutex,
};

pub struct File(Mutex<async_fs::File>);

impl File {
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self(Mutex::new(async_fs::File::open(path).await?)))
    }

    pub async fn read_exact_at(
        &self,
        buffer: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        let mut file = self.0.lock().await;
        file.seek(SeekFrom::Start(offset)).await?;
        file.read_exact(buffer).await
    }

    /// Unsupported on WASM.
    pub async fn to_sync(&self) -> io::Result<std::fs::File> {
        let file = self.0.lock().await;
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            Ok(file.as_fd().try_clone_to_owned()?.into())
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsHandle;
            Ok(file.as_handle().try_clone_to_owned()?.into())
        }
    }
}

pub fn read_exact_at(
    file: &std::fs::File,
    buffer: &mut [u8],
    offset: u64,
) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut buffer = buffer;
        let mut offset = offset;
        while !buffer.is_empty() {
            match file.seek_read(buffer, offset) {
                Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
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

pub async fn exists(path: impl AsRef<Path>) -> bool {
    async_fs::metadata(path).await.is_ok()
}
