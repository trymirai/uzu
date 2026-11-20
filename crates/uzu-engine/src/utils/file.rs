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
}

/// Tell the OS to avoid caching as much as possible while opening the file.
pub fn open_random_access(path: impl AsRef<Path>) -> io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // This is hard-coded FILE_FLAG_RANDOM_ACCESS from the Win32 headers.
        // I feel it's not worth it to pull a crate for this.
        options.custom_flags(0x1000_0000);
    }
    let file = options.open(path)?;
    #[cfg(target_vendor = "apple")]
    {
        use std::os::fd::AsRawFd;
        for (command, value) in [(libc::F_RDAHEAD, 0), (libc::F_NOCACHE, 1)] {
            if unsafe { libc::fcntl(file.as_raw_fd(), command, value) } == -1 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::fd::AsRawFd;
        let errno = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
        if errno != 0 {
            return Err(io::Error::from_raw_os_error(errno));
        }
    }
    Ok(file)
}

// TODO(perf): issue the reads concurrently.
pub fn read_exact_at(reads: Vec<(&std::fs::File, u64, &mut [u8])>) -> io::Result<()> {
    for (file, offset, buffer) in reads {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(buffer, offset)?;
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
        }
    }
    Ok(())
}

pub async fn exists(path: impl AsRef<Path>) -> bool {
    async_fs::metadata(path).await.is_ok()
}
