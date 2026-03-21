mod allocator;
mod range_allocator;

pub use allocator::{Allocation, AllocationPool, AllocationType, Allocator};
use range_allocator::{AllocationType as RangeAllocationType, RangeAllocator};
