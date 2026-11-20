use std::{io, path::Path, sync::OnceLock};

pub use async_fs::read;
use futures::{
    io::{AsyncReadExt, AsyncSeekExt, SeekFrom},
    lock::Mutex,
};
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

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

pub fn read_exact_at(reads: Vec<(&std::fs::File, u64, &mut [u8])>) -> io::Result<()> {
    // TODO(perf): This is bound by SSD latency, not CPU, so this should be
    // *mostly* fine. However, using io_uring on Linux or IoRing on Windows might
    // be the more proper way to do this.
    //
    // On macos there doesn't seem to be an equivalent. What we could do is load
    // it directly on the GPU, perhaps.
    io_pool().install(|| reads.into_par_iter().try_for_each(|(file, offset, buffer)| read_one(file, offset, buffer)))
}

const IO_THREADS: usize = 16;

fn io_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(IO_THREADS)
            .thread_name(|index| format!("uzu-io-{index}"))
            .build()
            .expect("failed to spawn the file read pool")
    })
}

fn read_one(
    file: &std::fs::File,
    offset: u64,
    buffer: &mut [u8],
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
