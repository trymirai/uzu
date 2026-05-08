use std::ops::Range;

#[derive(Clone, PartialEq)]
pub(super) struct MetalSparseHeapBufferMapping {
    gpu_address: u64,
    pages: Range<usize>,
}

impl MetalSparseHeapBufferMapping {
    pub fn new(
        gpu_address: u64,
        pages: Range<usize>,
    ) -> Self {
        Self {
            gpu_address,
            pages,
        }
    }

    pub fn gpu_address(&self) -> u64 {
        self.gpu_address
    }

    pub fn pages(&self) -> &Range<usize> {
        &self.pages
    }
}

pub(super) struct MetalSparseHeapMappingParameters {
    pub(super) buffer_pages: Range<usize>,
    pub(super) heap_page_offset: usize,
}
