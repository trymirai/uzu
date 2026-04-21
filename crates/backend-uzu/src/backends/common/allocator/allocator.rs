use std::{
    cell::RefCell,
    ops::Range,
    rc::{Rc, Weak},
};

use super::{RangeAllocationType, RangeAllocator};
use crate::backends::common::{Backend, Buffer, Context};

pub struct Allocation<B: Backend> {
    allocator: Rc<Allocator<B>>,
    buffer: Rc<RefCell<B::Buffer>>,
    range: Range<usize>,
    owner_range: Range<usize>,
    owner_token: Rc<()>,
}

impl<B: Backend> Allocation<B> {
    pub fn as_buffer_range<'a>(&'a self) -> (&'a B::Buffer, Range<usize>) {
        (unsafe { &*self.buffer.as_ptr() }, self.range.clone())
    }

    pub fn buffer(&self) -> Rc<RefCell<B::Buffer>> {
        self.buffer.clone()
    }

    pub fn view_at_offset(
        &self,
        byte_offset: usize,
        byte_len: usize,
    ) -> Self {
        let start = self.range.start + byte_offset;
        let end = start + byte_len;
        assert!(end <= self.range.end, "Allocation view exceeds backing range");
        Self {
            allocator: self.allocator.clone(),
            buffer: self.buffer.clone(),
            range: start..end,
            owner_range: self.owner_range.clone(),
            owner_token: self.owner_token.clone(),
        }
    }
}

impl<B: Backend> Clone for Allocation<B> {
    fn clone(&self) -> Self {
        Self {
            allocator: self.allocator.clone(),
            buffer: self.buffer.clone(),
            range: self.range.clone(),
            owner_range: self.owner_range.clone(),
            owner_token: self.owner_token.clone(),
        }
    }
}

impl<B: Backend> Drop for Allocation<B> {
    fn drop(&mut self) {
        if Rc::strong_count(&self.owner_token) == 1 {
            self.allocator.free(self.buffer.as_ptr() as *const B::Buffer, self.owner_range.clone())
        }
    }
}

pub struct AllocationPool<B: Backend> {
    reusable: bool,
    allocator: Rc<Allocator<B>>,
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
    buffer: Rc<RefCell<B::Buffer>>,
    range_allocator: RangeAllocator<usize>,
}

pub struct Allocator<B: Backend> {
    context: Weak<B::Context>,
    allocator_buffers: RefCell<Vec<AllocatorBuffer<B>>>,
    next_pool_number: RefCell<usize>,
    peak_memory_usage: RefCell<usize>,
}

impl<B: Backend> Allocator<B> {
    pub fn new(context: Weak<B::Context>) -> Rc<Self> {
        Rc::new(Self {
            context,
            allocator_buffers: RefCell::new(Vec::new()),
            next_pool_number: RefCell::new(0),
            peak_memory_usage: RefCell::new(0),
        })
    }

    pub fn allocate(
        self: &Rc<Self>,
        size: usize,
        allocation_type: AllocationType<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let allocation_size = size.max(1);
        let alignment = usize::clamp(allocation_size.next_power_of_two(), B::MIN_ALLOCATION_ALIGNMENT, 16_384);
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

        let mut allocator_buffers = self.allocator_buffers.borrow_mut();

        let mut found: Option<(Rc<RefCell<B::Buffer>>, Range<usize>)> = None;

        for allocator_buffer in allocator_buffers.iter_mut() {
            if let Some(range) = allocator_buffer.range_allocator.allocate_range_aligned(
                allocation_size,
                alignment,
                allocation_type.clone(),
            ) {
                found = Some((allocator_buffer.buffer.clone(), range));
                break;
            }
        }

        let (buffer, range) = if let Some((buffer, range)) = found {
            (buffer, range)
        } else {
            let new_allocator_buffer_size = if size == 0 {
                B::MIN_ALLOCATION_ALIGNMENT
            } else {
                usize::max(allocation_size, 268_435_456)
            };

            let mut allocator_buffer = AllocatorBuffer::<B> {
                buffer: Rc::new(RefCell::new(
                    self.context.upgrade().unwrap().create_buffer(new_allocator_buffer_size)?,
                )), // Upgrade can never fail
                range_allocator: RangeAllocator::new(0..new_allocator_buffer_size),
            };

            let buffer = allocator_buffer.buffer.clone();
            let range = allocator_buffer
                .range_allocator
                .allocate_range_aligned(allocation_size, alignment, allocation_type)
                .unwrap(); // Can never fail

            allocator_buffers.push(allocator_buffer);

            *self.peak_memory_usage.borrow_mut() =
                allocator_buffers.iter().map(|allocator_buffer| allocator_buffer.buffer.borrow().length()).sum();

            (buffer, range)
        };

        allocator_buffers.sort_by_key(|allocator_buffer| allocator_buffer.range_allocator.total_available());

        let owner_range = range.clone();
        let range = if size == 0 {
            range.start..range.start
        } else {
            range
        };

        Ok(Allocation {
            allocator: self.clone(),
            buffer,
            range,
            owner_range,
            owner_token: Rc::new(()),
        })
    }

    pub fn create_pool(
        self: &Rc<Self>,
        reusable: bool,
    ) -> AllocationPool<B> {
        let pool_number = *self.next_pool_number.borrow();

        *self.next_pool_number.borrow_mut() += 1;

        AllocationPool {
            reusable,
            allocator: self.clone(),
            pool_number,
        }
    }

    pub fn peak_memory_usage(self: Rc<Self>) -> usize {
        *self.peak_memory_usage.borrow()
    }

    // TODO: Maybe hysteresis in free/free_pool?

    fn free(
        self: &Rc<Self>,
        buffer: *const B::Buffer,
        range: Range<usize>,
    ) {
        let mut allocator_buffers = self.allocator_buffers.borrow_mut();

        let (allocator_buffer_index, allocator_buffer) = allocator_buffers
            .iter_mut()
            .enumerate()
            .find(|(_allocator_buffer_index, allocator_buffer)| {
                (allocator_buffer.buffer.as_ptr() as *const B::Buffer) == buffer
            })
            .unwrap(); // Can never fail

        allocator_buffer.range_allocator.free_range(range);

        if allocator_buffer.range_allocator.is_empty() {
            allocator_buffers.remove(allocator_buffer_index);
        }

        allocator_buffers.sort_by_key(|allocator_buffer| allocator_buffer.range_allocator.total_available());
    }

    fn free_pool(
        self: &Rc<Self>,
        pool: &AllocationPool<B>,
    ) {
        let mut allocator_buffers = self.allocator_buffers.borrow_mut();
        let mut free_idxs = Vec::new();

        for (allocation_buffer_idx, allocation_buffer) in allocator_buffers.iter_mut().enumerate() {
            allocation_buffer.range_allocator.free_pool(pool.pool_number);
            if allocation_buffer.range_allocator.is_empty() {
                free_idxs.push(allocation_buffer_idx);
            }
        }

        for (i, free_idx) in free_idxs.into_iter().enumerate() {
            allocator_buffers.remove(free_idx - i);
        }

        allocator_buffers.sort_by_key(|allocator_buffer| allocator_buffer.range_allocator.total_available());
    }
}
