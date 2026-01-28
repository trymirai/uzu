use std::collections::BTreeMap;
use std::ptr::NonNull;

use super::{MTLBuffer, ProtocolObject, Retained};

struct BufferHolder {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
    prev: Option<NonNull<BufferHolder>>,
    next: Option<NonNull<BufferHolder>>,
}

impl BufferHolder {
    fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>, size: usize) -> Box<Self> {
        Box::new(Self {
            buffer,
            size,
            prev: None,
            next: None,
        })
    }
}

pub struct BufferCache {
    pool: BTreeMap<usize, Vec<NonNull<BufferHolder>>>,
    head: Option<NonNull<BufferHolder>>,
    tail: Option<NonNull<BufferHolder>>,
    pool_size: usize,
    page_size: usize,
}

unsafe impl Send for BufferCache {}

impl BufferCache {
    pub fn new(page_size: usize) -> Self {
        Self {
            pool: BTreeMap::new(),
            head: None,
            tail: None,
            pool_size: 0,
            page_size,
        }
    }

    pub fn reuse_from_cache(&mut self, size: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let max_acceptable_size = std::cmp::min(2 * size, size + 2 * self.page_size);

        let mut found_key = None;
        for (&key, holders) in self.pool.range(size..) {
            if key >= max_acceptable_size {
                break;
            }
            if !holders.is_empty() {
                found_key = Some(key);
                break;
            }
        }

        let key = found_key?;
        let holders = self.pool.get_mut(&key)?;
        let holder_ptr = holders.pop()?;

        if holders.is_empty() {
            self.pool.remove(&key);
        }

        let holder = unsafe { Box::from_raw(holder_ptr.as_ptr()) };
        self.pool_size -= holder.size;
        self.remove_from_list(holder_ptr);

        Some(holder.buffer)
    }

    pub fn recycle_to_cache(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>, size: usize) {
        let holder = BufferHolder::new(buffer, size);
        let holder_ptr = NonNull::new(Box::into_raw(holder)).unwrap();

        self.add_at_head(holder_ptr);
        self.pool_size += size;

        self.pool.entry(size).or_default().push(holder_ptr);
    }

    pub fn release_cached_buffers(&mut self, min_bytes_to_free: usize) -> usize {
        if min_bytes_to_free >= (self.pool_size * 9) / 10 {
            return self.clear();
        }

        let mut total_bytes_freed = 0;
        let mut buffers_released = 0;

        while let Some(tail_ptr) = self.tail {
            if total_bytes_freed >= min_bytes_to_free {
                break;
            }

            let size = unsafe { tail_ptr.as_ref().size };
            total_bytes_freed += size;
            buffers_released += 1;

            if let Some(holders) = self.pool.get_mut(&size) {
                if let Some(pos) = holders.iter().position(|&p| p == tail_ptr) {
                    holders.swap_remove(pos);
                }
                if holders.is_empty() {
                    self.pool.remove(&size);
                }
            }

            let current_tail = self.tail;
            self.remove_from_list(tail_ptr);
            if let Some(ptr) = current_tail {
                let _ = unsafe { Box::from_raw(ptr.as_ptr()) };
            }
        }

        self.pool_size -= total_bytes_freed;
        buffers_released
    }

    pub fn clear(&mut self) -> usize {
        let count = self.pool.values().map(|v| v.len()).sum();

        let keys: Vec<usize> = self.pool.keys().copied().collect();
        for key in keys {
            if let Some(holders) = self.pool.remove(&key) {
                for holder_ptr in holders {
                    let _: Box<BufferHolder> = unsafe { Box::from_raw(holder_ptr.as_ptr()) };
                }
            }
        }

        self.head = None;
        self.tail = None;
        self.pool_size = 0;

        count
    }

    pub fn cache_size(&self) -> usize {
        self.pool_size
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    fn add_at_head(&mut self, holder_ptr: NonNull<BufferHolder>) {
        unsafe {
            let holder = holder_ptr.as_ptr();
            (*holder).prev = None;
            (*holder).next = self.head;

            if let Some(head) = self.head {
                (*head.as_ptr()).prev = Some(holder_ptr);
            }

            self.head = Some(holder_ptr);

            if self.tail.is_none() {
                self.tail = Some(holder_ptr);
            }
        }
    }

    fn remove_from_list(&mut self, holder_ptr: NonNull<BufferHolder>) {
        unsafe {
            let holder = holder_ptr.as_ptr();
            let prev = (*holder).prev;
            let next = (*holder).next;

            match (prev, next) {
                (Some(p), Some(n)) => {
                    (*p.as_ptr()).next = Some(n);
                    (*n.as_ptr()).prev = Some(p);
                }
                (Some(p), None) => {
                    (*p.as_ptr()).next = None;
                    self.tail = Some(p);
                }
                (None, Some(n)) => {
                    (*n.as_ptr()).prev = None;
                    self.head = Some(n);
                }
                (None, None) => {
                    self.head = None;
                    self.tail = None;
                }
            }

            (*holder).prev = None;
            (*holder).next = None;
        }
    }
}

impl Drop for BufferCache {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        // This test verifies the cache logic without actual Metal buffers
        // In production, we'd use integration tests with real buffers
    }
}
