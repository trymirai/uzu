use std::{ops::Range, rc::Rc};

use crate::backends::{
    common::{Backend, Context, SparseBuffer},
    metal::Metal,
};

fn create_context() -> Rc<<Metal as Backend>::Context> {
    <Metal as Backend>::Context::new().expect("Failed to create Metal context")
}

/// Capacity sized to fit `heap_count` worth of heaps.
fn buffer_capacity(
    ctx: &<Metal as Backend>::Context,
    heap_count: usize,
) -> usize {
    heap_count * ctx.sparse_heaps_mapper_mut().heap_capacity_bytes()
}

/// Buffer-page range covering `heap_count` worth of heaps starting at `heap_offset`.
fn pages_for_heaps(
    ctx: &<Metal as Backend>::Context,
    heap_offset: usize,
    heap_count: usize,
) -> Range<usize> {
    let pages_per_heap = ctx.sparse_heaps_mapper_mut().heap_capacity_pages();
    (heap_offset * pages_per_heap)..((heap_offset + heap_count) * pages_per_heap)
}

#[uzu_test]
fn test_mapping_succeeds() {
    let ctx = create_context();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
}

#[uzu_test]
fn test_unmapping_succeeds() {
    let ctx = create_context();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
    sparse_buffer.unmapping(ctx.as_ref(), &pages).expect("Failed to unmap sparse buffer");
}

#[uzu_test]
fn test_partial_unmapping_succeeds() {
    let ctx = create_context();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let mapped = pages_for_heaps(ctx.as_ref(), 0, 4);
    let unmapped = pages_for_heaps(ctx.as_ref(), 1, 2);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &mapped).expect("Failed to map sparse buffer");
    sparse_buffer.unmapping(ctx.as_ref(), &unmapped).expect("Failed to unmap sparse buffer");
}

#[uzu_test]
fn test_mapping_uses_minimum_heaps() {
    let ctx = create_context();
    let heap_count = 4;
    let capacity = buffer_capacity(ctx.as_ref(), heap_count);
    let pages = pages_for_heaps(ctx.as_ref(), 0, heap_count);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");

    let heaps = ctx.sparse_heaps_mapper_mut();
    let expected = pages.len().div_ceil(heaps.heap_capacity_pages());
    assert_eq!(heaps.heaps_count(), expected, "mapping should allocate the minimum number of heaps");
}

#[uzu_test]
fn test_remapping_reuses_freed_pages() {
    let ctx = create_context();
    let capacity = buffer_capacity(ctx.as_ref(), 6);
    let initial_pages = pages_for_heaps(ctx.as_ref(), 0, 4);
    let unmapped_pages = pages_for_heaps(ctx.as_ref(), 1, 1);
    let extra_pages = pages_for_heaps(ctx.as_ref(), 4, 1);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &initial_pages).expect("Failed to map sparse buffer");
    let initial_heaps = ctx.sparse_heaps_mapper_mut().heaps_count();

    sparse_buffer.unmapping(ctx.as_ref(), &unmapped_pages).expect("Failed to unmap sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &extra_pages).expect("Failed to remap sparse buffer");

    assert_eq!(
        ctx.sparse_heaps_mapper_mut().heaps_count(),
        initial_heaps,
        "remapping after unmap should reuse freed heap pages instead of allocating new heaps",
    );
}

#[uzu_test]
fn test_full_unmap_then_remap_reuses_heaps() {
    let ctx = create_context();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let pages = pages_for_heaps(ctx.as_ref(), 0, 4);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &pages).expect("Failed to map sparse buffer");
    let initial_heaps = ctx.sparse_heaps_mapper_mut().heaps_count();

    sparse_buffer.unmapping(ctx.as_ref(), &pages).expect("Failed to unmap sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &pages).expect("Failed to remap sparse buffer");

    assert_eq!(
        ctx.sparse_heaps_mapper_mut().heaps_count(),
        initial_heaps,
        "remapping the same range after a full unmap should not grow the heap pool",
    );
}

#[uzu_test]
fn test_sequential_mappings_are_compact() {
    let ctx = create_context();
    let capacity = buffer_capacity(ctx.as_ref(), 4);
    let first = pages_for_heaps(ctx.as_ref(), 0, 2);
    let second = pages_for_heaps(ctx.as_ref(), 2, 2);

    let mut sparse_buffer = ctx.create_sparse_buffer(capacity).expect("Failed to create sparse buffer");
    sparse_buffer.mapping(ctx.as_ref(), &first).expect("Failed to map first range");
    sparse_buffer.mapping(ctx.as_ref(), &second).expect("Failed to map second range");

    let heaps = ctx.sparse_heaps_mapper_mut();
    let total_pages = first.len() + second.len();
    let expected = total_pages.div_ceil(heaps.heap_capacity_pages());
    assert_eq!(
        heaps.heaps_count(),
        expected,
        "two adjacent mappings should pack into the same minimum number of heaps",
    );
}
