use std::{range::Range, sync::Arc};

struct BumpAllocatorPage<T> {
    page: Arc<T>,
    size: usize,
    offset: usize,
}

pub struct BumpAllocator<T> {
    min_page_size: usize,
    page_size_alignment: usize,
    pages: Vec<BumpAllocatorPage<T>>,
}

impl<T> BumpAllocator<T> {
    pub fn new(
        min_page_size: usize,
        page_size_alignment: usize,
    ) -> Self {
        Self {
            min_page_size,
            page_size_alignment,
            pages: Vec::new(),
        }
    }

    pub fn allocate<E>(
        &mut self,
        size: usize,
        alignment: usize,
        upstream: impl FnOnce(usize) -> Result<T, E>,
    ) -> Result<BumpAllocation<T>, E> {
        assert!(size > 0);

        if let Some(page) = self.pages.last_mut()
            && let page_offset_aligned = page.offset.next_multiple_of(alignment)
            && page_offset_aligned + size <= page.size
        {
            let allocation = BumpAllocation {
                page: page.page.clone(),
                range: (page_offset_aligned..page_offset_aligned + size).into(),
            };
            page.offset = page_offset_aligned + size;
            return Ok(allocation);
        }

        let new_page_size = usize::max(size, self.min_page_size).next_multiple_of(self.page_size_alignment);
        let new_page = Arc::new(upstream(new_page_size)?);
        self.pages.push(BumpAllocatorPage {
            page: new_page.clone(),
            size: new_page_size,
            offset: size,
        });

        Ok(BumpAllocation {
            page: new_page,
            range: (0..size).into(),
        })
    }

    pub fn is_done(&self) -> bool {
        self.pages.iter().all(|page| Arc::strong_count(&page.page) == 1)
    }
}

pub struct BumpAllocation<T> {
    page: Arc<T>,
    range: Range<usize>,
}

impl<T> BumpAllocation<T> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}
