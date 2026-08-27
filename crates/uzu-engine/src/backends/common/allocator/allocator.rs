use std::{
    fmt::Debug,
    range::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use parking_lot::Mutex;

use crate::backends::common::{
    Backend, Buffer,
    allocator::{RangeAllocationType, RangeAllocator},
};

pub trait Storage: Debug + Send + Sync + 'static {
    type Backend: Backend;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_ALLOCATION_ALIGNMENT: usize;
    const ALLOCATION_GRANULARITY: usize;
}

pub struct Allocation<S: Storage> {
    allocator: Arc<Allocator<S>>,
    buffer: Arc<S>,
    range: Range<usize>,
    allocation_type: RangeAllocationType,
}

impl<S: Storage> Allocation<S> {
    pub(in crate::backends) fn buffer(&self) -> &S {
        &self.buffer
    }

    pub(in crate::backends) fn offset(&self) -> usize {
        self.range.start
    }
}

impl<S: Storage> std::fmt::Debug for Allocation<S> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("Allocation").field("buffer", &self.buffer).field("range", &self.range).finish_non_exhaustive()
    }
}

impl<S: Storage> Buffer for Allocation<S> {
    type Backend = S::Backend;

    fn size(&self) -> usize {
        self.range.iter().len()
    }
}

impl<S: Storage> Drop for Allocation<S> {
    fn drop(&mut self) {
        self.allocator.free(self)
    }
}

pub struct AllocationPool<S: Storage> {
    allocator: Arc<Allocator<S>>,
    pool_number: usize,
}

impl<S: Storage> Drop for AllocationPool<S> {
    fn drop(&mut self) {
        self.allocator.free_pool(self)
    }
}

pub enum AllocationType<'a, S: Storage> {
    Global,
    Pooled {
        pool: &'a AllocationPool<S>,
        cpu_available: bool,
    },
}

struct AllocatorBuffer<S: Storage> {
    buffer: Arc<S>,
    range_allocator: RangeAllocator,
}

pub struct Allocator<S: Storage> {
    create_storage: Box<dyn Fn(usize) -> Result<S, <S::Backend as Backend>::Error> + Send + Sync>,
    allocator_buffers: Mutex<Vec<AllocatorBuffer<S>>>,
    next_pool_number: AtomicUsize,
}

impl<S: Storage> Allocator<S> {
    pub(in crate::backends) fn new(
        create_storage: impl Fn(usize) -> Result<S, <S::Backend as Backend>::Error> + Send + Sync + 'static
    ) -> Arc<Self> {
        Arc::new(Self {
            create_storage: Box::new(create_storage),
            allocator_buffers: Mutex::new(Vec::new()),
            next_pool_number: AtomicUsize::new(0),
        })
    }

    pub(in crate::backends) fn allocate(
        self: &Arc<Self>,
        size: usize,
        allocation_type: AllocationType<S>,
    ) -> Result<Allocation<S>, <S::Backend as Backend>::Error> {
        assert!(size > 0, "allocation size must be greater than 0");
        let alignment =
            usize::clamp(size.next_power_of_two(), S::MIN_ALLOCATION_ALIGNMENT, S::MAX_ALLOCATION_ALIGNMENT);
        let allocation_type = match allocation_type {
            AllocationType::Global => RangeAllocationType::Global,
            AllocationType::Pooled {
                pool,
                cpu_available,
            } => RangeAllocationType::Pooled {
                pool: pool.pool_number,
                can_alias_before: !cpu_available,
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
            let new_allocator_buffer_size = usize::max(size, S::ALLOCATION_GRANULARITY);

            let mut allocator_buffer = AllocatorBuffer::<S> {
                buffer: Arc::new((self.create_storage)(new_allocator_buffer_size)?),
                range_allocator: RangeAllocator::new((0..new_allocator_buffer_size).into()),
            };

            let buffer = allocator_buffer.buffer.clone();
            let range =
                allocator_buffer.range_allocator.allocate_range_aligned(size, alignment, allocation_type).unwrap(); // Can never fail

            allocator_buffers.push(allocator_buffer);
            let allocator_buffer_index = allocator_buffers.len() - 1;
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);

            (buffer, range)
        };

        Ok(Allocation {
            allocator: self.clone(),
            buffer,
            range,
            allocation_type,
        })
    }

    pub(in crate::backends) fn create_pool(self: &Arc<Self>) -> AllocationPool<S> {
        let pool_number = self.next_pool_number.fetch_add(1, Ordering::Relaxed);

        AllocationPool {
            allocator: self.clone(),
            pool_number,
        }
    }

    // TODO: Maybe hysteresis in free/free_pool?

    fn free(
        self: &Arc<Self>,
        allocation: &Allocation<S>,
    ) {
        let mut allocator_buffers = self.allocator_buffers.lock();

        let allocator_buffer_index = allocator_buffers
            .iter()
            .position(|allocator_buffer| Arc::ptr_eq(&allocator_buffer.buffer, &allocation.buffer))
            .unwrap(); // Can never fail

        allocator_buffers[allocator_buffer_index]
            .range_allocator
            .free_range(allocation.range, allocation.allocation_type);

        if allocator_buffers[allocator_buffer_index].range_allocator.is_empty() {
            allocator_buffers.remove(allocator_buffer_index);
        } else {
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);
        }
    }

    fn free_pool(
        self: &Arc<Self>,
        pool: &AllocationPool<S>,
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
        allocator_buffers: &mut [AllocatorBuffer<S>],
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
