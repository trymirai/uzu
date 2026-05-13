use std::rc::Rc;

use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    backends::{
        common::{Backend, Context},
        metal::{Metal, metal_extensions::SparsePageSizeExt},
    },
    prelude::MetalContext,
};

fn create_context() -> Rc<MetalContext> {
    <Metal as Backend>::Context::new().expect("Failed to create Metal context")
}

fn create_sparse_buffer(
    ctx: &MetalContext,
    capacity_bytes: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let page_size = ctx.sparse_heap_pool().page_size();
    let aligned = capacity_bytes.next_multiple_of(page_size.in_bytes());
    ctx.device
        .new_buffer_with_length_options_placement_sparse_page_size(
            aligned,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        )
        .expect("Failed to create sparse buffer")
}

#[test]
fn test_new_pool_is_empty() {
    let ctx = create_context();
    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_map_empty_range_allocates_no_heaps() {
    let ctx = create_context();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..0)).expect("map empty range");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_map_single_heap_allocates_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..pages_per_heap)).expect("map");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_map_multi_heap_allocates_minimum_heaps() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 4 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..4 * pages_per_heap)).expect("map");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 4);
}

#[test]
fn test_map_partial_heap_rounds_up_heap_count() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..pages_per_heap + 1)).expect("map");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_sequential_mappings_pack_into_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..half)).expect("map first");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(half..pages_per_heap)).expect("map second");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_map_overflows_existing_heap_into_new_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..half)).expect("map first");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(half..half + pages_per_heap)).expect("map overflow");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_two_buffers_share_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..half)).expect("map b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_full_unmap_removes_heaps() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);
    let pages = 0..2 * pages_per_heap;

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &pages).expect("map");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer, &pages).expect("unmap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_partial_unmap_keeps_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..pages_per_heap)).expect("map");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer, &(0..pages_per_heap / 2)).expect("partial unmap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_with_other_buffer_keeps_mappings() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);
    let pages = 0..pages_per_heap;

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &pages).expect("map a");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_b, &pages).expect("unmap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_one_buffer_keeps_heap_for_other() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..half)).expect("map b");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_a, &(0..half)).expect("unmap a");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_remap_after_unmap_does_not_grow_pool() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);
    let pages = 0..2 * pages_per_heap;

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &pages).expect("map");
    let initial = ctx.sparse_heap_pool().heaps_count();
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer, &pages).expect("unmap");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &pages).expect("remap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), initial);
}

#[test]
fn test_remap_into_freed_gap_reuses_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..pages_per_heap)).expect("map full");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer, &(0..half)).expect("unmap half");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..half)).expect("remap into gap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_sequential_partial_unmaps_release_full_heap() {
    // Regression: after a partial unmap splits a rangemap entry, the surviving
    // entry must still report the correct buffer↔heap correspondence so that a
    // subsequent unmap of the remaining suffix targets the right heap pages.
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer, &(0..pages_per_heap)).expect("map");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer, &(0..half)).expect("unmap prefix");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer, &(half..pages_per_heap)).expect("unmap suffix");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_three_buffers_share_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_c = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..third)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..third)).expect("map b");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_c, &(0..third)).expect("map c");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_two_buffers_exactly_fill_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..pages_per_heap - half)).expect("map b fills heap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_third_buffer_spills_when_heap_filled_by_others() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_c = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..pages_per_heap - half)).expect("map b fills heap");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_c, &(0..1)).expect("map c spills");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_unmap_buffer_frees_pages_for_another_in_same_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_c = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..pages_per_heap - half)).expect("map b fills heap");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_a, &(0..half)).expect("unmap a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_c, &(0..half)).expect("map c into gap");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_middle_of_three_buffers_keeps_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_c = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..third)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..third)).expect("map b");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_c, &(0..third)).expect("map c");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_b, &(0..third)).expect("unmap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_all_shared_buffers_releases_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_c = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..third)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..third)).expect("map b");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_c, &(0..third)).expect("map c");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_a, &(0..third)).expect("unmap a");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_b, &(0..third)).expect("unmap b");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_c, &(0..third)).expect("unmap c");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_second_buffer_spans_existing_and_new_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..pages_per_heap)).expect("map b spans heaps");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_unmap_spanning_buffer_keeps_heap_with_other_buffer() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..pages_per_heap)).expect("map b spans heaps");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_b, &(0..pages_per_heap)).expect("unmap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_remap_two_buffers_after_unmap_does_not_grow_pool() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let buffer_a = create_sparse_buffer(ctx.as_ref(), cap);
    let buffer_b = create_sparse_buffer(ctx.as_ref(), cap);

    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..half)).expect("map b");
    let initial = ctx.sparse_heap_pool().heaps_count();
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_a, &(0..half)).expect("unmap a");
    ctx.sparse_heap_pool_mut().unmap(ctx.as_ref(), &buffer_b, &(0..half)).expect("unmap b");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_a, &(0..half)).expect("remap a");
    ctx.sparse_heap_pool_mut().map(ctx.as_ref(), &buffer_b, &(0..half)).expect("remap b");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), initial);
}

#[test]
fn test_heap_capacity_pages_matches_byte_capacity() {
    let ctx = create_context();
    let pool = ctx.sparse_heap_pool();
    let page_size_bytes = pool.page_size().in_bytes();

    assert_eq!(pool.heap_capacity_pages(), pool.heap_capacity_bytes() / page_size_bytes);
}
