use std::{
    ffi::c_void,
    ops::Range,
    ptr::NonNull,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use bytemuck::{AnyBitPattern, NoUninit};
use memmap2::MmapMut;
use parking_lot::Mutex;

use crate::backends::common::{
    AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, BufferRangeMut, BufferRangeRef, Context, DenseBuffer,
    allocator::{RangeAllocationType, RangeAllocator},
};

pub struct Allocation<B: Backend> {
    allocator: Arc<Allocator<B>>,
    buffer: Arc<B::DenseBuffer>,
    range: Range<usize>,
    kind: AllocationKind,
}

/// Where an allocation's bytes live.
enum AllocationKind {
    /// A range of one of the allocator's buffers, returned to it on drop.
    Ranged(RangeAllocationType),
    /// A range of a file mapping, used in place. Held only to keep the mapping alive as long as its allocations; the
    /// field drops after `buffer`, so no buffer outlives the memory under it.
    Mapped {
        _mapping: Arc<MmapMut>,
    },
}

impl<B: Backend> Allocation<B> {
    pub fn size(&self) -> usize {
        self.range.len()
    }

    pub fn as_slice_mut<T: NoUninit + AnyBitPattern>(&mut self) -> &mut [T] {
        let buffer_range = self.as_buffer_range_mut();
        let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
        let bytes = unsafe {
            std::slice::from_raw_parts_mut((buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start), range.len())
        };
        bytemuck::cast_slice_mut(bytes)
    }

    pub fn copyin<T: NoUninit + AnyBitPattern>(
        &mut self,
        data: &[T],
    ) {
        self.as_slice_mut().copy_from_slice(data);
    }

    pub fn as_slice<T: AnyBitPattern>(&self) -> &[T] {
        let buffer_range = self.as_buffer_range_ref();
        let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
        let bytes = unsafe {
            std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start), range.len())
        };
        bytemuck::cast_slice(bytes)
    }

    pub fn copyout<T: AnyBitPattern>(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<B: Backend> AsBufferRangeRef for Allocation<B> {
    type Buffer = B::DenseBuffer;

    fn as_buffer_range_ref<'a>(&'a self) -> BufferRangeRef<'a, B::DenseBuffer> {
        BufferRangeRef::new(self.buffer.as_ref(), self.range.clone())
    }
}

impl<B: Backend> AsBufferRangeMut for Allocation<B> {
    fn as_buffer_range_mut<'a>(&'a mut self) -> BufferRangeMut<'a, B::DenseBuffer> {
        // SAFETY: allocator algorithm (hopefully if there is no bugs) guarantees no two overlapping live allocations can exist (which is the contract of BufferRangeMut)
        unsafe { BufferRangeMut::new_shared(self.buffer.as_ref(), self.range.clone()) }
    }
}

impl<B: Backend> Drop for Allocation<B> {
    fn drop(&mut self) {
        if let AllocationKind::Ranged(allocation_type) = self.kind {
            self.allocator.free(&self.buffer, self.range.clone(), allocation_type)
        }
    }
}

/// A file mapped copy-on-write and used in place: its tensors become allocations of its byte ranges, wrapped without
/// copying into buffers of at most the backend's maximum buffer length, each created the first time a range needs it.
/// Clean mapped pages are file-backed, so they do not count as the process's own memory.
pub struct MappedFile<B: Backend> {
    allocator: Arc<Allocator<B>>,
    // declared before the map so the buffers over it are released before it can be unmapped
    windows: Mutex<Vec<(Range<usize>, Arc<B::DenseBuffer>)>>,
    map: Arc<MmapMut>,
}

impl<B: Backend> MappedFile<B> {
    pub fn file_len(&self) -> usize {
        self.map.len()
    }

    /// The mapped length: the file rounded up to whole pages (the tail of the last page reads as zeros).
    pub fn mapped_len(&self) -> usize {
        self.map.len().div_ceil(rustix::param::page_size()) * rustix::param::page_size()
    }

    /// The file's bytes `range`, in place.
    pub fn allocation(
        &self,
        range: Range<usize>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(range.end <= self.mapped_len());
        let mut windows = self.windows.lock();
        let found = windows.iter().find(|(window, _)| window.start <= range.start && range.end <= window.end);
        let (start, buffer) = match found {
            Some((window, buffer)) => (window.start, buffer.clone()),
            None => {
                let context = self.allocator.context.upgrade().unwrap(); // the allocator never outlives its context
                let page = rustix::param::page_size();
                let start = range.start / page * page;
                let end = start + usize::min(context.max_buffer_length() / page * page, self.mapped_len() - start);
                assert!(range.end <= end, "a {} byte range does not fit one buffer", range.len());
                // SAFETY: page-aligned, inside the mapping, and the mapping outlives the buffer (see AllocationKind::Mapped)
                let pointer = NonNull::new(unsafe { self.map.as_ptr().add(start) } as *mut c_void).unwrap();
                let buffer = Arc::new(unsafe { context.create_buffer_over_host_memory(pointer, end - start)? });
                windows.push((start..end, buffer.clone()));
                (start, buffer)
            },
        };
        Ok(Allocation {
            allocator: self.allocator.clone(),
            buffer,
            range: range.start - start..range.end - start,
            kind: AllocationKind::Mapped {
                _mapping: self.map.clone(),
            },
        })
    }
}

pub struct AllocationPool<B: Backend> {
    reusable: bool,
    allocator: Arc<Allocator<B>>,
    pool_number: usize,
}

impl<B: Backend> Drop for AllocationPool<B> {
    fn drop(&mut self) {
        self.allocator.free_pool(self)
    }
}

pub enum AllocationType<'a, B: Backend> {
    Global,
    Pooled {
        pool: &'a AllocationPool<B>,
        cpu_available: bool,
    },
}

struct AllocatorBuffer<B: Backend> {
    buffer: Arc<B::DenseBuffer>,
    range_allocator: RangeAllocator,
}

pub struct Allocator<B: Backend> {
    context: Weak<B::Context>,
    allocator_buffers: Mutex<Vec<AllocatorBuffer<B>>>,
    next_pool_number: AtomicUsize,
    peak_memory_usage: AtomicUsize,
}

impl<B: Backend> Allocator<B> {
    pub fn new(context: Weak<B::Context>) -> Arc<Self> {
        Arc::new(Self {
            context,
            allocator_buffers: Mutex::new(Vec::new()),
            next_pool_number: AtomicUsize::new(0),
            peak_memory_usage: AtomicUsize::new(0),
        })
    }

    pub fn allocate(
        self: &Arc<Self>,
        size: usize,
        allocation_type: AllocationType<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(size > 0, "allocation size must be greater than 0");
        let alignment =
            usize::clamp(size.next_power_of_two(), B::MIN_ALLOCATION_ALIGNMENT, B::MAX_ALLOCATION_ALIGNMENT);
        let allocation_type = match allocation_type {
            AllocationType::Global => RangeAllocationType::Global,
            AllocationType::Pooled {
                pool,
                cpu_available,
            } => RangeAllocationType::Pooled {
                pool: pool.pool_number,
                can_alias_before: !cpu_available,
                can_alias_after: !(cpu_available && pool.reusable),
            },
        };

        let mut allocator_buffers = self.allocator_buffers.lock();

        let found = allocator_buffers.iter_mut().enumerate().find_map(|(allocator_buffer_index, allocator_buffer)| {
            let range = allocator_buffer.range_allocator.allocate_range_aligned(size, alignment, allocation_type)?;

            Some((allocator_buffer_index, allocator_buffer.buffer.clone(), range))
        });

        let (buffer, range) = if let Some((allocator_buffer_index, buffer, range)) = found {
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);

            (buffer, range)
        } else {
            let new_allocator_buffer_size = usize::max(size, B::ALLOCATION_GRANULARITY);

            let mut allocator_buffer = AllocatorBuffer::<B> {
                buffer: Arc::new(self.context.upgrade().unwrap().create_buffer(new_allocator_buffer_size)?), // Upgrade can never fail
                range_allocator: RangeAllocator::new(0..new_allocator_buffer_size),
            };

            let buffer = allocator_buffer.buffer.clone();
            let range =
                allocator_buffer.range_allocator.allocate_range_aligned(size, alignment, allocation_type).unwrap(); // Can never fail

            allocator_buffers.push(allocator_buffer);
            let allocator_buffer_index = allocator_buffers.len() - 1;
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);

            self.peak_memory_usage.store(
                allocator_buffers.iter().map(|allocator_buffer| allocator_buffer.buffer.size()).sum(),
                Ordering::Relaxed,
            );

            (buffer, range)
        };

        Ok(Allocation {
            allocator: self.clone(),
            buffer,
            range,
            kind: AllocationKind::Ranged(allocation_type),
        })
    }

    pub fn create_pool(
        self: &Arc<Self>,
        reusable: bool,
    ) -> AllocationPool<B> {
        let pool_number = self.next_pool_number.fetch_add(1, Ordering::Relaxed);

        AllocationPool {
            reusable,
            allocator: self.clone(),
            pool_number,
        }
    }

    pub fn peak_memory_usage(&self) -> usize {
        self.peak_memory_usage.load(Ordering::Relaxed)
    }

    // TODO: Maybe hysteresis in free/free_pool?

    pub fn map_file(
        self: &Arc<Self>,
        map: MmapMut,
    ) -> MappedFile<B> {
        MappedFile {
            allocator: self.clone(),
            map: Arc::new(map),
            windows: Mutex::new(Vec::new()),
        }
    }

    fn free(
        self: &Arc<Self>,
        buffer: &Arc<B::DenseBuffer>,
        range: Range<usize>,
        allocation_type: RangeAllocationType,
    ) {
        let mut allocator_buffers = self.allocator_buffers.lock();

        let allocator_buffer_index = allocator_buffers
            .iter()
            .position(|allocator_buffer| Arc::ptr_eq(&allocator_buffer.buffer, buffer))
            .unwrap(); // Can never fail

        allocator_buffers[allocator_buffer_index].range_allocator.free_range(range, allocation_type);

        if allocator_buffers[allocator_buffer_index].range_allocator.is_empty() {
            allocator_buffers.remove(allocator_buffer_index);
        } else {
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);
        }
    }

    fn free_pool(
        self: &Arc<Self>,
        pool: &AllocationPool<B>,
    ) {
        let mut allocator_buffers = self.allocator_buffers.lock();

        allocator_buffers.retain_mut(|allocation_buffer| {
            allocation_buffer.range_allocator.free_pool(pool.pool_number);
            !allocation_buffer.range_allocator.is_empty()
        });

        if allocator_buffers.len() > 1 {
            allocator_buffers.sort_by_key(|allocator_buffer| allocator_buffer.range_allocator.total_available());
        }
    }

    fn restore_buffer_order(
        allocator_buffers: &mut [AllocatorBuffer<B>],
        mut index: usize,
    ) {
        while index > 0
            && allocator_buffers[index].range_allocator.total_available()
                < allocator_buffers[index - 1].range_allocator.total_available()
        {
            allocator_buffers.swap(index, index - 1);
            index -= 1;
        }

        while index + 1 < allocator_buffers.len()
            && allocator_buffers[index].range_allocator.total_available()
                > allocator_buffers[index + 1].range_allocator.total_available()
        {
            allocator_buffers.swap(index, index + 1);
            index += 1;
        }
    }
}

#[cfg(all(test, backend = "metal"))]
#[path = "../../../../tests/unit/backends/common/allocator/allocator.rs"]
mod tests;
