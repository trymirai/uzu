use test_macros::uzu_test;

use crate::{
    backends::{
        common::SparseBuffer,
        metal::{Metal, metal_extensions::SparsePageSizeExt},
    },
    common::helpers::{create_context, sparse_buffer_create},
};

#[uzu_test]
fn test_new_pool_is_empty() {
    let ctx = create_context::<Metal>();
    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[uzu_test]
fn test_map_empty_range_allocates_no_heaps() {
    let ctx = create_context::<Metal>();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..0)).expect("map empty range");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[uzu_test]
fn test_map_single_heap_allocates_one_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..pages_per_heap)).expect("map");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_map_multi_heap_allocates_minimum_heaps() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 4 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..4 * pages_per_heap)).expect("map");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 4);
}

#[uzu_test]
fn test_map_partial_heap_rounds_up_heap_count() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..pages_per_heap + 1)).expect("map");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[uzu_test]
fn test_sequential_mappings_pack_into_one_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..half)).expect("map first");
    sparse_buffer.map(&ctx, &(half..pages_per_heap)).expect("map second");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_map_overflows_existing_heap_into_new_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..half)).expect("map first");
    sparse_buffer.map(&ctx, &(half..half + pages_per_heap)).expect("map overflow");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[uzu_test]
fn test_two_buffers_share_one_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..half)).expect("map b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_full_unmap_removes_heaps() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let pages = 0..2 * pages_per_heap;

    sparse_buffer.map(&ctx, &pages).expect("map");
    sparse_buffer.unmap(&ctx, &pages).expect("unmap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[uzu_test]
fn test_partial_unmap_keeps_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..pages_per_heap)).expect("map");
    sparse_buffer.unmap(&ctx, &(0..pages_per_heap / 2)).expect("partial unmap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_unmap_with_other_buffer_keeps_mappings() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let pages = 0..pages_per_heap;

    sparse_buffer_a.map(&ctx, &pages).expect("map a");
    sparse_buffer_b.unmap(&ctx, &pages).expect("unmap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_unmap_one_buffer_keeps_heap_for_other() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..half)).expect("map b");
    sparse_buffer_a.unmap(&ctx, &(0..half)).expect("unmap a");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_remap_after_unmap_does_not_grow_pool() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let pages = 0..2 * pages_per_heap;

    sparse_buffer.map(&ctx, &pages).expect("map");
    let initial = ctx.sparse_heap_pool().heaps_count();
    sparse_buffer.unmap(&ctx, &pages).expect("unmap");
    sparse_buffer.map(&ctx, &pages).expect("remap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), initial);
}

#[uzu_test]
fn test_remap_into_freed_gap_reuses_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..pages_per_heap)).expect("map full");
    sparse_buffer.unmap(&ctx, &(0..half)).expect("unmap half");
    sparse_buffer.map(&ctx, &(0..half)).expect("remap into gap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_sequential_partial_unmaps_release_full_heap() {
    // Regression: after a partial unmap splits a rangemap entry, the surviving
    // entry must still report the correct buffer↔heap correspondence so that a
    // subsequent unmap of the remaining suffix targets the right heap pages.
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer.map(&ctx, &(0..pages_per_heap)).expect("map");
    sparse_buffer.unmap(&ctx, &(0..half)).expect("unmap prefix");
    sparse_buffer.unmap(&ctx, &(half..pages_per_heap)).expect("unmap suffix");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[uzu_test]
fn test_three_buffers_share_one_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_c = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..third)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..third)).expect("map b");
    sparse_buffer_c.map(&ctx, &(0..third)).expect("map c");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_two_buffers_exactly_fill_one_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..pages_per_heap - half)).expect("map b fills heap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_third_buffer_spills_when_heap_filled_by_others() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_c = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..pages_per_heap - half)).expect("map b fills heap");
    sparse_buffer_c.map(&ctx, &(0..1)).expect("map c spills");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[uzu_test]
fn test_unmap_buffer_frees_pages_for_another_in_same_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_c = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..pages_per_heap - half)).expect("map b fills heap");
    sparse_buffer_a.unmap(&ctx, &(0..half)).expect("unmap a");
    sparse_buffer_c.map(&ctx, &(0..half)).expect("map c into gap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_unmap_middle_of_three_buffers_keeps_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_c = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..third)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..third)).expect("map b");
    sparse_buffer_c.map(&ctx, &(0..third)).expect("map c");
    sparse_buffer_b.unmap(&ctx, &(0..third)).expect("unmap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_unmap_all_shared_buffers_releases_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_c = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..third)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..third)).expect("map b");
    sparse_buffer_c.map(&ctx, &(0..third)).expect("map c");
    sparse_buffer_a.unmap(&ctx, &(0..third)).expect("unmap a");
    sparse_buffer_b.unmap(&ctx, &(0..third)).expect("unmap b");
    sparse_buffer_c.unmap(&ctx, &(0..third)).expect("unmap c");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[uzu_test]
fn test_second_buffer_spans_existing_and_new_heap() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..pages_per_heap)).expect("map b spans heaps");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[uzu_test]
fn test_unmap_spanning_buffer_keeps_heap_with_other_buffer() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..pages_per_heap)).expect("map b spans heaps");
    sparse_buffer_b.unmap(&ctx, &(0..pages_per_heap)).expect("unmap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[uzu_test]
fn test_remap_two_buffers_after_unmap_does_not_grow_pool() {
    let ctx = create_context::<Metal>();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let mut sparse_buffer_a = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);
    let mut sparse_buffer_b = sparse_buffer_create::<Metal>(ctx.as_ref(), cap);

    sparse_buffer_a.map(&ctx, &(0..half)).expect("map a");
    sparse_buffer_b.map(&ctx, &(0..half)).expect("map b");
    let initial = ctx.sparse_heap_pool().heaps_count();
    sparse_buffer_a.unmap(&ctx, &(0..half)).expect("unmap a");
    sparse_buffer_b.unmap(&ctx, &(0..half)).expect("unmap b");
    sparse_buffer_a.map(&ctx, &(0..half)).expect("remap a");
    sparse_buffer_b.map(&ctx, &(0..half)).expect("remap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), initial);
}

#[uzu_test]
fn test_heap_capacity_pages_matches_byte_capacity() {
    let ctx = create_context::<Metal>();
    let pool = ctx.sparse_heap_pool();
    let page_size_bytes = pool.page_size().in_bytes();

    assert_eq!(pool.heap_capacity_pages(), pool.heap_capacity_bytes() / page_size_bytes);
}
