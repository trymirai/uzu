use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, binary_heap, hash_map::Entry},
    iter::once,
    range::Range,
    sync::{
        Arc,
        atomic::{self, AtomicUsize},
    },
};

use metal::{MTLDevice, MTLDeviceExt, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLSparsePageSize, MTLStorageMode};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;
use rangemap::RangeSet;

use crate::backends::metal::{error::MetalError, metal_extensions::SparsePageSizeExt};

struct MetalHeapsInnerNonfull {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    num_free_pages: usize,
}

impl PartialEq for MetalHeapsInnerNonfull {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for MetalHeapsInnerNonfull {}
impl PartialOrd for MetalHeapsInnerNonfull {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MetalHeapsInnerNonfull {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.num_free_pages
            .cmp(&other.num_free_pages)
            .reverse()
            .then(Retained::as_ptr(&self.heap).cmp(&Retained::as_ptr(&other.heap)))
    }
}

struct MetalHeapsInner {
    free_ranges: HashMap<usize, RangeSet<usize>>,
    nonfull_heaps: BinaryHeap<MetalHeapsInnerNonfull>,
}

pub struct MetalHeaps {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    peak_memory_usage: Arc<AtomicUsize>,
    inner: Mutex<MetalHeapsInner>,
    page_size: MTLSparsePageSize,
    pages_per_heap: usize,
}

impl MetalHeaps {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        peak_memory_usage: Arc<AtomicUsize>,
        page_size: MTLSparsePageSize,
        pages_per_heap: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            peak_memory_usage,
            inner: Mutex::new(MetalHeapsInner {
                free_ranges: HashMap::new(),
                nonfull_heaps: BinaryHeap::new(),
            }),
            page_size,
            pages_per_heap,
        })
    }

    pub fn page_size(&self) -> MTLSparsePageSize {
        self.page_size
    }

    pub fn allocate(
        self: &Arc<Self>,
        number: usize,
    ) -> Result<Vec<MetalHeapPage>, MetalError> {
        let mut remaining = number;
        let mut batch = Vec::new();

        let mut inner_locked = self.inner.lock();
        let inner = &mut *inner_locked;

        while remaining > 0
            && let Some(mut nonfull_heap) = inner.nonfull_heaps.peek_mut()
        {
            let heap_ptr = Retained::as_ptr(&nonfull_heap.heap) as usize;
            let free_ranges = inner.free_ranges.get_mut(&heap_ptr).unwrap();

            while remaining > 0
                && let Some(free_range) = free_ranges.first().cloned()
            {
                let used_range = free_range.start..usize::min(free_range.end, free_range.start + remaining);
                batch.push(MetalHeapPage {
                    heaps: self.clone(),
                    heap: nonfull_heap.heap.clone(),
                    range: used_range.clone().into(),
                });
                nonfull_heap.num_free_pages -= used_range.len();
                remaining -= used_range.len();
                free_ranges.remove(used_range);
            }

            if free_ranges.is_empty() {
                inner.free_ranges.remove(&heap_ptr);
                binary_heap::PeekMut::pop(nonfull_heap);
            }
        }

        while remaining > 0 {
            let descriptor = MTLHeapDescriptor::new();
            descriptor.set_type(MTLHeapType::Placement);
            descriptor.set_storage_mode(MTLStorageMode::Private);
            descriptor.set_size(self.pages_per_heap * self.page_size.in_bytes());
            descriptor.set_sparse_page_size(self.page_size);
            descriptor.set_max_compatible_placement_sparse_page_size(self.page_size);
            let heap = self.device.new_heap_with_descriptor(&descriptor).ok_or(MetalError::CannotCreateHeap)?;

            self.peak_memory_usage.fetch_max(self.device.current_allocated_size(), atomic::Ordering::Relaxed);

            let allocated_pages = usize::min(remaining, self.pages_per_heap);

            batch.push(MetalHeapPage {
                heaps: self.clone(),
                heap: heap.clone(),
                range: (0..allocated_pages).into(),
            });

            if allocated_pages < self.pages_per_heap {
                inner
                    .free_ranges
                    .insert(Retained::as_ptr(&heap) as usize, once(allocated_pages..self.pages_per_heap).collect());
                inner.nonfull_heaps.push(MetalHeapsInnerNonfull {
                    heap,
                    num_free_pages: self.pages_per_heap - allocated_pages,
                });
            }

            remaining -= allocated_pages;
        }

        Ok(batch)
    }

    fn free(
        &self,
        heap: &Retained<ProtocolObject<dyn MTLHeap>>,
        range: Range<usize>,
    ) {
        let mut inner_locked = self.inner.lock();
        let inner = &mut *inner_locked;

        match inner.free_ranges.entry(Retained::as_ptr(heap) as usize) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().insert(range.into());
                let mut nonfull = std::mem::take(&mut inner.nonfull_heaps).into_vec();
                let nonfull_idx = nonfull
                    .iter()
                    .enumerate()
                    .find(|(_, h)| (Retained::as_ptr(&h.heap) as usize) == *entry.key())
                    .unwrap()
                    .0;
                if entry.get().first().unwrap() == &(0..self.pages_per_heap) {
                    entry.remove();
                    nonfull.swap_remove(nonfull_idx);
                } else {
                    nonfull[nonfull_idx].num_free_pages += range.iter().len();
                }
                inner.nonfull_heaps.extend(nonfull);
            },
            Entry::Vacant(entry) => {
                if range != (0..self.pages_per_heap).into() {
                    entry.insert(once(range.into()).collect());
                    inner.nonfull_heaps.push(MetalHeapsInnerNonfull {
                        heap: heap.clone(),
                        num_free_pages: range.iter().len(),
                    });
                }
            },
        }
    }
}

pub struct MetalHeapPage {
    heaps: Arc<MetalHeaps>,
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    range: Range<usize>,
}

impl MetalHeapPage {
    pub fn heap(&self) -> &Retained<ProtocolObject<dyn MTLHeap>> {
        &self.heap
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl Drop for MetalHeapPage {
    fn drop(&mut self) {
        self.heaps.free(&self.heap, self.range);
    }
}
