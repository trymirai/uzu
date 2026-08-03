use std::{ops::Range, sync::Arc};

use metal::{MTL4UpdateSparseBufferMappingOperation, MTLBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::metal::sparse::sparse_heap::MetalSparseHeap;

#[derive(Clone, PartialEq)]
pub(crate) struct MetalSparseHeapBufferMapping {
    // Anchors for an originally inserted contiguous mapping. They survive
    // rangemap splits unchanged, so the heap↔buffer correspondence can always
    // be recovered from any surviving heap subrange via `buffer_pages_for`.
    heap_page_anchor: usize,
    buffer_page_anchor: usize,
}

impl MetalSparseHeapBufferMapping {
    pub fn new(
        heap_page_anchor: usize,
        buffer_page_anchor: usize,
    ) -> Self {
        Self {
            heap_page_anchor,
            buffer_page_anchor,
        }
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

pub(crate) struct MetalSparseMappingOpsBatch {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub heap: Arc<Mutex<MetalSparseHeap>>,
    pub mtl_operations: Box<[MTL4UpdateSparseBufferMappingOperation]>,
}
