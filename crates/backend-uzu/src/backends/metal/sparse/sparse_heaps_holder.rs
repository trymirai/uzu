use std::{cmp::min, ops::Range};

use metal::{MTLBuffer, MTLSparsePageSize};
use objc2::runtime::ProtocolObject;

use crate::{
    backends::metal::{
        error::MetalError,
        metal_extensions::SparsePageSizeExt,
        sparse::sparse_heap::{MetalSparseHeap, MetalSparseHeapOperationParameters},
    },
    prelude::MetalContext,
};

pub struct MetalSparseHeapsHolder {
    heaps: Vec<MetalSparseHeap>,
    heap_capacity_bytes: usize,
    page_size: MTLSparsePageSize,
}

impl MetalSparseHeapsHolder {
    pub fn new() -> Result<Self, MetalError> {
        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = 16 * 4 * page_size.byte_size().as_u64() as usize;
        Ok(Self {
            heaps: vec![],
            page_size,
            heap_capacity_bytes: heap_capacity,
        })
    }

    pub fn mapping(
        &mut self,
        context: &MetalContext,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        buffer_pages: &Range<usize>,
    ) -> Result<(), MetalError> {
        let heap_capacity_pages = self.heap_capacity_pages();
        let mut remaining_buffer_pages = buffer_pages.clone();

        // try to find pages in existing heaps
        for heap_index in 0..self.heaps.len() {
            if remaining_buffer_pages.start >= remaining_buffer_pages.end {
                break;
            }

            let heap_range = 0..heap_capacity_pages;
            let gaps: Vec<Range<usize>> = self.heaps[heap_index].mapped_pages().gaps(&heap_range).collect();

            let mut ops: Vec<MetalSparseHeapOperationParameters> = Vec::new();
            for gap in gaps {
                if remaining_buffer_pages.start >= remaining_buffer_pages.end {
                    break;
                }

                let map_pages_count =
                    min(gap.end - gap.start, remaining_buffer_pages.end - remaining_buffer_pages.start);
                let buffer_pages = Range {
                    start: remaining_buffer_pages.start,
                    end: remaining_buffer_pages.start + map_pages_count,
                };
                ops.push(MetalSparseHeapOperationParameters {
                    buffer_pages,
                    heap_page_offset: gap.start,
                });
                remaining_buffer_pages.start += map_pages_count;
            }

            self.heaps[heap_index].execute(buffer, &context.command_queue4, &ops, true);
        }

        // allocate new heaps
        let new_heaps_required = if remaining_buffer_pages.start >= remaining_buffer_pages.end {
            0
        } else {
            (remaining_buffer_pages.end - remaining_buffer_pages.start).div_ceil(heap_capacity_pages)
        };
        for _ in 0..new_heaps_required {
            let mut sparse_heap = MetalSparseHeap::new(context, self.heap_capacity_bytes, self.page_size)?;

            let map_pages_count = min(heap_capacity_pages, remaining_buffer_pages.len());
            let buffer_pages = Range {
                start: remaining_buffer_pages.start,
                end: remaining_buffer_pages.start + map_pages_count,
            };
            let op_params = MetalSparseHeapOperationParameters {
                buffer_pages,
                heap_page_offset: 0,
            };
            sparse_heap.execute(buffer, &context.command_queue4, &[op_params], true);
            remaining_buffer_pages.start += map_pages_count;

            self.heaps.push(sparse_heap);
        }

        Ok(())
    }

    pub fn unmapping(
        &mut self,
        context: &MetalContext,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        buffer_pages: &Range<usize>,
    ) -> Result<(), MetalError> {
        let buffer_address = buffer.gpu_address();
        for heap in &mut self.heaps {
            let mut ops_params: Vec<MetalSparseHeapOperationParameters> = vec![];
            for (heap_range, (mapped_address, buffer_range)) in heap.mapped_pages().iter() {
                if *mapped_address != buffer_address {
                    continue;
                }

                let overlap_start = buffer_pages.start.max(buffer_range.start);
                let overlap_end = buffer_pages.end.min(buffer_range.end);
                if overlap_start >= overlap_end {
                    continue;
                }

                let offset_within_range = overlap_start - buffer_range.start;
                ops_params.push(MetalSparseHeapOperationParameters {
                    buffer_pages: overlap_start..overlap_end,
                    heap_page_offset: heap_range.start + offset_within_range,
                });
            }
            heap.execute(buffer, &context.command_queue4, &ops_params, false);
        }

        Ok(())
    }

    pub fn page_size(&self) -> MTLSparsePageSize {
        self.page_size
    }

    pub fn heaps_count(&self) -> usize {
        self.heaps.len()
    }

    pub fn heap_capacity_bytes(&self) -> usize {
        self.heap_capacity_bytes
    }

    pub fn heap_capacity_pages(&self) -> usize {
        self.heap_capacity_bytes / self.page_size.byte_size().as_u64() as usize
    }
}
