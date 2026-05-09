use std::ops::Range;

#[derive(Clone, PartialEq)]
pub(super) struct MetalSparseHeapBufferMapping {
    gpu_address: u64,
    // Anchors for an originally inserted contiguous mapping. They survive
    // rangemap splits unchanged, so the heap↔buffer correspondence can always
    // be recovered from any surviving heap subrange via `buffer_pages_for`.
    heap_page_anchor: usize,
    buffer_page_anchor: usize,
}

impl MetalSparseHeapBufferMapping {
    pub fn new(
        gpu_address: u64,
        heap_page_anchor: usize,
        buffer_page_anchor: usize,
    ) -> Self {
        Self {
            gpu_address,
            heap_page_anchor,
            buffer_page_anchor,
        }
    }

    pub fn gpu_address(&self) -> u64 {
        self.gpu_address
    }

    pub fn buffer_pages_for(
        &self,
        heap_range: &Range<usize>,
    ) -> Range<usize> {
        let start = self.buffer_page_anchor + (heap_range.start - self.heap_page_anchor);
        let end = self.buffer_page_anchor + (heap_range.end - self.heap_page_anchor);
        start..end
    }
}

pub(super) struct MetalSparseHeapMappingParameters {
    pub(super) buffer_pages: Range<usize>,
    pub(super) heap_page_offset: usize,
}
