use std::{collections::VecDeque, mem::ManuallyDrop, range::Range, sync::Arc};

use parking_lot::Mutex;

pub struct PoolAllocator<T> {
    pages: [Mutex<VecDeque<T>>; 32],
}

impl<T> PoolAllocator<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            pages: Default::default(),
        })
    }

    pub fn allocate<E>(
        self: &Arc<Self>,
        size: usize,
        upstream: impl FnOnce(usize) -> Result<T, E>,
    ) -> Result<PoolAllocation<T>, E> {
        assert!(size > 0 && size <= 2 * 1024 * 1024 * 1024);

        let page = if let Some(page) = self.pages[(size - 1).bit_width() as usize].lock().pop_front() {
            page
        } else {
            upstream(size.next_power_of_two())?
        };

        Ok(PoolAllocation {
            allocator: self.clone(),
            page: ManuallyDrop::new(page),
            range: (0..size).into(),
        })
    }

    fn alias(
        self: &Arc<Self>,
        page: T,
        range: Range<usize>,
    ) {
        self.pages[(range.iter().len() - 1).bit_width() as usize].lock().push_back(page);
    }
}

pub struct PoolAllocation<T> {
    allocator: Arc<PoolAllocator<T>>,
    page: ManuallyDrop<T>,
    range: Range<usize>,
}

impl<T> PoolAllocation<T> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl<T> Drop for PoolAllocation<T> {
    fn drop(&mut self) {
        self.allocator.alias(unsafe { ManuallyDrop::take(&mut self.page) }, self.range);
    }
}
