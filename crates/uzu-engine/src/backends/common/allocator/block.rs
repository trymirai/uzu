use std::{collections::HashMap, mem::ManuallyDrop, range::Range, sync::Arc};

use parking_lot::Mutex;

// TODO: this is a stub, real block allocator should suballocate from large pages

pub struct BlockAllocator<T> {
    min_page_size: usize,
    cache: Mutex<HashMap<usize, Vec<T>>>,
}

impl<T> BlockAllocator<T> {
    pub fn new(min_page_size: usize) -> Arc<Self> {
        Arc::new(Self {
            min_page_size,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn allocate<E>(
        self: &Arc<Self>,
        size: usize,
        upstream: impl FnOnce(usize) -> Result<T, E>,
    ) -> Result<BlockAllocation<T>, E> {
        assert!(size > 0);

        let page = if let mut cache_locked = self.cache.lock()
            && let Some(cache_entries) = cache_locked.get_mut(&size.max(self.min_page_size))
            && let Some(cache_entry) = cache_entries.pop()
        {
            cache_entry
        } else {
            upstream(size.max(self.min_page_size))?
        };

        Ok(BlockAllocation {
            allocator: self.clone(),
            page: ManuallyDrop::new(page),
            range: (0..size).into(),
        })
    }

    fn free(
        self: &Arc<Self>,
        page: T,
        range: Range<usize>,
    ) {
        self.cache.lock().entry(range.iter().len().max(self.min_page_size)).or_default().push(page);
    }
}

pub struct BlockAllocation<T> {
    allocator: Arc<BlockAllocator<T>>,
    page: ManuallyDrop<T>,
    range: Range<usize>,
}

impl<T> BlockAllocation<T> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl<T> Drop for BlockAllocation<T> {
    fn drop(&mut self) {
        self.allocator.free(unsafe { ManuallyDrop::take(&mut self.page) }, self.range);
    }
}
