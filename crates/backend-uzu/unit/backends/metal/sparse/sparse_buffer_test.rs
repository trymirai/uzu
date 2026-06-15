use std::ops::Range;

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, SparseBuffer},
        metal::Metal,
    },
    common::helpers::{create_context, sparse_buffer_create},
};

/// Capacity sized to fit `heap_count` worth of heaps.
fn buffer_capacity(
    ctx: &<Metal as Backend>::Context,
    heap_count: usize,
) -> usize {
    heap_count * ctx.sparse_heap_pool().heap_capacity_bytes()
}

/// Buffer-page range covering `heap_count` worth of heaps starting at `heap_offset`.
fn pages_for_heaps(
    ctx: &<Metal as Backend>::Context,
    heap_offset: usize,
    heap_count: usize,
) -> Range<usize> {
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    (heap_offset * pages_per_heap)..((heap_offset + heap_count) * pages_per_heap)
}

#[uzu_test]
fn test_mapping_succeeds() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
}

#[uzu_test]
fn test_unmapping_succeeds() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
    sparse_buffer.unmap(ctx.as_ref(), &pages).expect("Failed to unmap sparse buffer");
}

#[uzu_test]
fn test_partial_unmapping_succeeds() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let mapped = pages_for_heaps(ctx.as_ref(), 0, 4);
    let unmapped = pages_for_heaps(ctx.as_ref(), 1, 2);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &mapped).expect("Failed to map sparse buffer");
    sparse_buffer.unmap(ctx.as_ref(), &unmapped).expect("Failed to unmap sparse buffer");
}

#[uzu_test]
fn test_mapping_uses_minimum_heaps() {
    let ctx = create_context::<Metal>();
    let heap_count = 4;
    let capacity = buffer_capacity(ctx.as_ref(), heap_count);
    let pages = pages_for_heaps(ctx.as_ref(), 0, heap_count);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");

    let heaps = ctx.sparse_heap_pool();
    let expected = pages.len().div_ceil(heaps.heap_capacity_pages());
    assert_eq!(heaps.heaps_count(), expected, "mapping should allocate the minimum number of heaps");
}

#[uzu_test]
fn test_remapping_reuses_freed_pages() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 6);
    let initial_pages = pages_for_heaps(ctx.as_ref(), 0, 4);
    let unmapped_pages = pages_for_heaps(ctx.as_ref(), 1, 1);
    let extra_pages = pages_for_heaps(ctx.as_ref(), 4, 1);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &initial_pages).expect("Failed to map sparse buffer");
    let initial_heaps = ctx.sparse_heap_pool().heaps_count();

    sparse_buffer.unmap(ctx.as_ref(), &unmapped_pages).expect("Failed to unmap sparse buffer");
    sparse_buffer.map(ctx.as_ref(), &extra_pages).expect("Failed to remap sparse buffer");

    assert_eq!(
        ctx.sparse_heap_pool().heaps_count(),
        initial_heaps,
        "remapping after unmap should reuse freed heap pages instead of allocating new heaps",
    );
}

#[uzu_test]
fn test_full_unmap_then_remap_reuses_heaps() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
    let initial_heaps = ctx.sparse_heap_pool().heaps_count();

    sparse_buffer.unmap(ctx.as_ref(), &pages).expect("Failed to unmap sparse buffer");
    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to remap sparse buffer");

    assert_eq!(
        ctx.sparse_heap_pool().heaps_count(),
        initial_heaps,
        "remapping the same range after a full unmap should not grow the heap pool",
    );
}

#[uzu_test]
fn test_remapping_same_pages_is_noop() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
    let initial_heaps = ctx.sparse_heap_pool().heaps_count();

    sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to remap sparse buffer");

    assert_eq!(
        ctx.sparse_heap_pool().heaps_count(),
        initial_heaps,
        "remapping the same pages should not allocate new heaps",
    );
}

#[uzu_test]
fn test_mapping_multiple_gaps_reserves_pages_between_gaps() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let capacity = buffer_capacity(ctx.as_ref(), 2);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &(1..2)).expect("Failed to map initial page");
    sparse_buffer.map(ctx.as_ref(), &(0..pages_per_heap + 1)).expect("Failed to map range with multiple gaps");

    assert_eq!(
        ctx.sparse_heap_pool().heaps_count(),
        2,
        "mapping multiple gaps in one call must reserve heap pages between gaps",
    );
}

#[uzu_test]
fn test_drop_releases_pool_heaps() {
    // Regression: a mapped buffer dropped without an explicit unmap must
    // still release its heap pages back to the shared pool.
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 2);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 2);

    {
        let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
        sparse_buffer.map(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
        assert!(ctx.sparse_heap_pool().heaps_count() > 0);
    }

    assert_eq!(
        ctx.sparse_heap_pool().heaps_count(),
        0,
        "dropping a mapped sparse buffer should release its heap pages",
    );
}

#[uzu_test]
fn test_drop_does_not_disturb_other_buffer_mappings() {
    // Dropping one buffer must not unmap heap pages held by another buffer
    // sharing the same pool.
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 1);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 1);

    let mut keeper = sparse_buffer_create::<Metal>(&ctx, capacity);
    keeper.map(ctx.as_ref(), &pages).expect("Failed to map keeper buffer");
    let baseline = ctx.sparse_heap_pool().heaps_count();

    {
        let mut transient = sparse_buffer_create::<Metal>(&ctx, capacity);
        transient.map(ctx.as_ref(), &pages).expect("Failed to map transient buffer");
    }

    let current_heaps = ctx.sparse_heap_pool().heaps_count();
    assert_eq!(current_heaps, baseline, "dropping a transient buffer must leave keeper buffer's mappings intact",);
}

#[uzu_test]
fn test_sequential_mappings_are_compact() {
    let ctx = create_context::<Metal>();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let first = pages_for_heaps(ctx.as_ref(), 0, 2);
    let second = pages_for_heaps(ctx.as_ref(), 2, 2);

    let mut sparse_buffer = sparse_buffer_create::<Metal>(&ctx, capacity);
    sparse_buffer.map(ctx.as_ref(), &first).expect("Failed to map first range");
    sparse_buffer.map(ctx.as_ref(), &second).expect("Failed to map second range");

    let heaps = ctx.sparse_heap_pool();
    let total_pages = first.len() + second.len();
    let expected = total_pages.div_ceil(heaps.heap_capacity_pages());
    assert_eq!(
        heaps.heaps_count(),
        expected,
        "two adjacent mappings should pack into the same minimum number of heaps",
    );
}
