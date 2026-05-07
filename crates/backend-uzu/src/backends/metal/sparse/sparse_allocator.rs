use std::{cmp::min, ops::Range};

use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeMap;

use crate::{
    backends::metal::{error::MetalError, metal_extensions::SparsePageSizeExt},
    prelude::MetalContext,
};

struct MetalSparseHeapOperationParameters {
    buffer_pages: Range<usize>,
    heap_page_offset: usize,
}

struct MetalSparseHeap {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    mapped_pages: RangeMap<usize, ()>,
}

impl MetalSparseHeap {
    pub fn new(
        context: &MetalContext,
        capacity_bytes: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let page_size_bytes = page_size.byte_size().as_u64() as usize;
        let aligned_capacity = capacity_bytes.div_ceil(page_size_bytes) * page_size_bytes;

        let heap_desc = MTLHeapDescriptor::new();
        // Sparse buffers must be backed by a Placement heap with a sparse page size set;
        // `MTLHeapType::Sparse` is for sparse textures and trips a runtime assertion when
        // passed to updateBufferMappings.
        heap_desc.set_type(MTLHeapType::Placement);
        heap_desc.set_storage_mode(MTLStorageMode::Private);
        heap_desc.set_size(aligned_capacity);
        heap_desc.set_max_compatible_placement_sparse_page_size(page_size);
        let heap = context
            .device
            .new_heap_with_descriptor(&heap_desc)
            .ok_or_else(|| MetalError::SparseHeapAlloc(aligned_capacity, page_size.byte_size()))?;
        Ok(Self {
            heap,
            mapped_pages: RangeMap::new(),
        })
    }

    pub fn mapped_pages(&self) -> &RangeMap<usize, ()> {
        &self.mapped_pages
    }

    pub fn execute(
        &mut self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        cmd_queue: &ProtocolObject<dyn MTL4CommandQueue>,
        ops_parameters: &[MetalSparseHeapOperationParameters],
        mapping: bool,
    ) {
        if ops_parameters.is_empty() {
            return;
        }

        let operations = ops_parameters
            .iter()
            .map(|op| {
                let mode = if mapping {
                    MTLSparseTextureMappingMode::Map
                } else {
                    MTLSparseTextureMappingMode::Unmap
                };
                let buffer_range = Range {
                    start: op.buffer_pages.start,
                    end: op.buffer_pages.end,
                };
                MTL4UpdateSparseBufferMappingOperation::new(mode, buffer_range, op.heap_page_offset)
            })
            .collect::<Vec<_>>();

        cmd_queue.update_buffer_mappings(buffer, Some(&self.heap), &operations);

        for op in &operations {
            let heap_range = Range {
                start: op.heap_offset,
                end: op.heap_offset + op.buffer_range().len(),
            };
            if mapping {
                self.mapped_pages.insert(heap_range, ());
            } else {
                self.mapped_pages.remove(heap_range);
            }
        }
    }
}

pub struct MetalSparseHeapsHolder {
    heaps: Vec<MetalSparseHeap>,
    heap_capacity_bytes: usize,
    page_size: MTLSparsePageSize,
}

impl MetalSparseHeapsHolder {
    pub fn new() -> Result<Self, MetalError> {
        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = 1 * page_size.byte_size().as_u64() as usize;
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
        pages: &Range<usize>,
    ) -> Result<(), MetalError> {
        let heap_capacity_pages = self.heap_capacity_pages();
        let mut remaining_buffer_pages = pages.clone();

        // try to find pages in existing heaps
        for heap_index in 0..self.heaps.len() {
            if remaining_buffer_pages.start >= remaining_buffer_pages.end {
                break;
            }

            let heap_range = 0..heap_capacity_pages;
            let gaps: Vec<Range<usize>> = self.heaps[heap_index].mapped_pages.gaps(&heap_range).collect();

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
        pages: &Range<usize>,
    ) -> Result<(), MetalError> {
        Ok(())
    }

    pub fn page_size(&self) -> MTLSparsePageSize {
        self.page_size
    }

    fn heap_capacity_pages(&self) -> usize {
        self.heap_capacity_bytes / self.page_size.byte_size().as_u64() as usize
    }
}
