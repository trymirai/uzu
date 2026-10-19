use std::{fs::File, io, os::fd::AsRawFd, ptr::NonNull};

/// A file mapped copy-on-write. Buffers over it use the file's pages in place: clean pages are file-backed, so they
/// stay outside the process footprint.
pub struct FileMapping {
    pointer: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is plain memory that lives until drop.
unsafe impl Send for FileMapping {}
unsafe impl Sync for FileMapping {}

impl FileMapping {
    pub fn new(file: &File) -> io::Result<Self> {
        let len = file.metadata()?.len() as usize;
        // SAFETY: a fresh private mapping of the whole file; nothing else refers to it.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            pointer: NonNull::new(pointer.cast()).expect("mmap never returns null on success"),
            len,
        })
    }

    fn page_size() -> usize {
        // SAFETY: sysconf has no preconditions.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    /// Whole pages around `offset..offset + size` (the tail of the last page reads as zeros): their first byte's
    /// address, their length, and `offset` within them.
    pub fn pages(
        &self,
        offset: usize,
        size: usize,
    ) -> (NonNull<u8>, usize, usize) {
        assert!(offset + size <= self.len);
        let page = Self::page_size();
        let start = offset / page * page;
        // SAFETY: start is inside the mapping.
        (unsafe { self.pointer.add(start) }, (offset + size).next_multiple_of(page) - start, offset - start)
    }
}

impl Drop for FileMapping {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly what new mapped; every buffer over it holds an Arc to it until Metal releases it.
        unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.len) };
    }
}
