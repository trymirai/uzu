use proc_macros::uzu_test;

use super::{RangeAllocationType, RangeAllocator};
use crate::backends::common::{AllocationType, Backend, Context};

#[cfg(backend = "cpu")]
type TestBackend = crate::backends::cpu::Cpu;
#[cfg(all(backend = "metal", not(backend = "cpu")))]
type TestBackend = crate::backends::metal::Metal;

const POOL_CYCLES: usize = 1024;

#[uzu_test]
fn pool_numbers_are_recycled() {
    let context = <TestBackend as Backend>::Context::new().expect("context");

    let mut highest = 0usize;
    for _ in 0..POOL_CYCLES {
        let pool = context.create_allocation_pool(false);
        highest = highest.max(pool.pool_number());

        let allocation = context
            .create_allocation(
                4096,
                AllocationType::Pooled {
                    pool: &pool,
                    cpu_available: true,
                },
            )
            .expect("allocation");
        drop(allocation);
        drop(pool);
    }

    assert_eq!(highest, 0, "sequential pools must reuse the same number, got up to {highest}");
}

#[uzu_test]
fn nested_pools_stay_dense() {
    let context = <TestBackend as Backend>::Context::new().expect("context");

    for _ in 0..POOL_CYCLES {
        let outer = context.create_allocation_pool(false);
        let inner = context.create_allocation_pool(false);
        assert!(outer.pool_number() < 2 && inner.pool_number() < 2, "expected numbers 0 and 1");
        assert_ne!(outer.pool_number(), inner.pool_number(), "live pools must not share a number");
    }
}

fn pooled(pool: usize) -> RangeAllocationType {
    RangeAllocationType::Pooled {
        pool,
        can_alias_before: false,
        can_alias_after: false,
    }
}

fn cycle(
    allocator: &mut RangeAllocator,
    pool: usize,
) {
    let range = allocator.allocate_range_aligned(1024, 64, pooled(pool)).expect("range");
    allocator.free_range(range, pooled(pool));
    allocator.free_pool(pool);
}

/// Shows why the numbers matter: a monotonic counter grows per-pool state without bound.
#[uzu_test]
fn range_allocator_state_follows_pool_numbers() {
    let mut recycled = RangeAllocator::new(0..1 << 20);
    for _ in 0..POOL_CYCLES {
        cycle(&mut recycled, 0);
    }
    assert_eq!(recycled.pool_slot_count(), 1, "recycled numbers keep per-pool state at one slot");

    let mut growing = RangeAllocator::new(0..1 << 20);
    for pool in 0..POOL_CYCLES {
        cycle(&mut growing, pool);
    }
    assert_eq!(growing.pool_slot_count(), POOL_CYCLES, "distinct numbers grow per-pool state");
}
