use std::{
    collections::VecDeque,
    io,
    path::Path,
    sync::{
        Arc, Once, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

pub use async_fs::read;
use futures::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use parking_lot::{Condvar, Mutex};

pub struct File(futures::lock::Mutex<async_fs::File>);

impl File {
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self(futures::lock::Mutex::new(async_fs::File::open(path).await?)))
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
    // TODO(perf): This is bound by SSD latency, not CPU, so threads are *mostly* fine. However,
    // using io_uring on Linux or IoRing on Windows might be the more proper way to do this. On
    // macOS, Metal IO (`MTLIOCommandQueue`) seems to be slower (~2-5x) for small random reads.
    if reads.is_empty() {
        return Ok(());
    }
    let job = Arc::new(ReadJob::new(reads));
    io_pool().submit(job.clone());
    // Wait to make sure that borrowed buffers to the pool are sound
    job.wait()
}

// TODO(perf): The best count depends on the storage device and on how many reads a call has (fewer
// threads are cheaper for single-token decode, more are faster for prefill), so it should probably
// be tuned per machine and batch size. 32 is just a (reasonable) default.
const IO_THREADS: usize = 32;

// Not rayon, because rayon spins before sleeping.
struct IoPool {
    queue: Mutex<VecDeque<Arc<ReadJob>>>,
    wake: Condvar,
}

fn io_pool() -> &'static IoPool {
    static POOL: OnceLock<IoPool> = OnceLock::new();
    static STARTED: Once = Once::new();
    let pool = POOL.get_or_init(|| IoPool {
        queue: Mutex::new(VecDeque::new()),
        wake: Condvar::new(),
    });
    STARTED.call_once(|| {
        for index in 0..IO_THREADS {
            std::thread::Builder::new()
                .name(format!("uzu-io-{index}"))
                .spawn(|| pool.work())
                .expect("failed to spawn the file read pool");
        }
    });
    pool
}

impl IoPool {
    fn submit(
        &self,
        job: Arc<ReadJob>,
    ) {
        self.queue.lock().push_back(job);
        self.wake.notify_all();
    }

    fn work(&self) {
        loop {
            let job = {
                let mut queue = self.queue.lock();
                loop {
                    if let Some(job) = queue.front() {
                        break job.clone();
                    }
                    self.wake.wait(&mut queue);
                }
            };
            // Every thread works on the oldest job until its reads are all
            // claimed, so jobs finish in submission order.
            while let Some(index) = job.claim() {
                job.run(index);
            }
            let mut queue = self.queue.lock();
            if queue.front().is_some_and(|front| Arc::ptr_eq(front, &job)) {
                queue.pop_front();
            }
        }
    }
}

struct Read {
    file: *const std::fs::File,
    offset: u64,
    buffer: *mut u8,
    len: usize,
}

/// One [`read_exact_at`] call, with erased borrows (so that pool threads can hold it).
struct ReadJob {
    reads: Box<[Read]>,
    next: AtomicUsize,
    remaining: AtomicUsize,
    failed: AtomicBool,
    error: Mutex<Option<io::Error>>,
    done: Mutex<bool>,
    done_signal: Condvar,
}

// SAFETY: `read_exact_at` keeps the files and buffers borrowed until every read
// has finished, each read touches a different buffer, and `std::fs::File` is `Sync`.
unsafe impl Send for ReadJob {}
unsafe impl Sync for ReadJob {}

impl ReadJob {
    fn new(reads: Vec<(&std::fs::File, u64, &mut [u8])>) -> Self {
        let reads = reads
            .into_iter()
            .map(|(file, offset, buffer)| Read {
                file,
                offset,
                buffer: buffer.as_mut_ptr(),
                len: buffer.len(),
            })
            .collect::<Box<[_]>>();
        Self {
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(reads.len()),
            reads,
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
            done: Mutex::new(false),
            done_signal: Condvar::new(),
        }
    }

    fn claim(&self) -> Option<usize> {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        (index < self.reads.len()).then_some(index)
    }

    fn run(
        &self,
        index: usize,
    ) {
        // Once a read failed the call fails anyway, so skip the rest.
        if !self.failed.load(Ordering::Relaxed) {
            let read = &self.reads[index];
            // SAFETY: see `unsafe impl Sync for ReadJob`, `index` was claimed by this thread only.
            let (file, buffer) = unsafe { (&*read.file, std::slice::from_raw_parts_mut(read.buffer, read.len)) };
            if let Err(error) = read_one(file, read.offset, buffer) {
                self.failed.store(true, Ordering::Relaxed);
                self.error.lock().get_or_insert(error);
            }
        }
        // The buffer writes happen before this release, and the last decrement
        // acquires all of them before waking the caller through `done`.
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            *self.done.lock() = true;
            self.done_signal.notify_one();
        }
    }

    fn wait(&self) -> io::Result<()> {
        let mut done = self.done.lock();
        while !*done {
            self.done_signal.wait(&mut done);
        }
        drop(done);
        self.error.lock().take().map_or(Ok(()), Err)
    }
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

#[cfg(test)]
#[path = "../../unit/utils/file_test.rs"]
mod tests;
