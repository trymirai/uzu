#[path = "../../../../common/mod.rs"]
mod common;

use std::rc::Rc;

use test_tag::tag;

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

#[test]
fn test_new_pool_is_empty() {
    let ctx = create_context();
    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_map_empty_range_allocates_no_heaps() {
    let ctx = create_context();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..0)).expect("map empty range");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_map_single_heap_allocates_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..pages_per_heap)).expect("map");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_map_multi_heap_allocates_minimum_heaps() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 4 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..4 * pages_per_heap)).expect("map");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 4);
}

#[test]
fn test_map_partial_heap_rounds_up_heap_count() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..pages_per_heap + 1)).expect("map");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_sequential_mappings_pack_into_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..half)).expect("map first");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(half..pages_per_heap)).expect("map second");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_map_overflows_existing_heap_into_new_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..half)).expect("map first");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(half..half + pages_per_heap)).expect("map overflow");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_two_buffers_share_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..half)).expect("map b");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_full_unmap_removes_heaps() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let pages = 0..2 * pages_per_heap;

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &pages).expect("map");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse.mtl_buffer(), &pages).expect("unmap");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_partial_unmap_keeps_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..pages_per_heap)).expect("map");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse.mtl_buffer(), &(0..pages_per_heap / 2)).expect("partial unmap");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_with_other_buffer_keeps_mappings() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let pages = 0..pages_per_heap;

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &pages).expect("map a");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_b.mtl_buffer(), &pages).expect("unmap b");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_one_buffer_keeps_heap_for_other() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..half)).expect("map b");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("unmap a");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_remap_after_unmap_does_not_grow_pool() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let pages = 0..2 * pages_per_heap;

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &pages).expect("map");
    let initial = ctx.sparse_heap_pool().heaps_count();
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse.mtl_buffer(), &pages).expect("unmap");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &pages).expect("remap");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), initial);
}

#[test]
fn test_remap_into_freed_gap_reuses_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..pages_per_heap)).expect("map full");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse.mtl_buffer(), &(0..half)).expect("unmap half");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..half)).expect("remap into gap");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

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
    let sparse = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse.mtl_buffer(), &(0..pages_per_heap)).expect("map");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse.mtl_buffer(), &(0..half)).expect("unmap prefix");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse.mtl_buffer(), &(half..pages_per_heap)).expect("unmap suffix");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_three_buffers_share_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_c = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..third)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..third)).expect("map b");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_c.mtl_buffer(), &(0..third)).expect("map c");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_two_buffers_exactly_fill_one_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut()
        .map(&ctx, &sparse_b.mtl_buffer(), &(0..pages_per_heap - half))
        .expect("map b fills heap");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_third_buffer_spills_when_heap_filled_by_others() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_c = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut()
        .map(&ctx, &sparse_b.mtl_buffer(), &(0..pages_per_heap - half))
        .expect("map b fills heap");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_c.mtl_buffer(), &(0..1)).expect("map c spills");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_unmap_buffer_frees_pages_for_another_in_same_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_c = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut()
        .map(&ctx, &sparse_b.mtl_buffer(), &(0..pages_per_heap - half))
        .expect("map b fills heap");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("unmap a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_c.mtl_buffer(), &(0..half)).expect("map c into gap");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_middle_of_three_buffers_keeps_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_c = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..third)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..third)).expect("map b");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_c.mtl_buffer(), &(0..third)).expect("map c");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_b.mtl_buffer(), &(0..third)).expect("unmap b");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_unmap_all_shared_buffers_releases_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let third = pages_per_heap / 3;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_c = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..third)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..third)).expect("map b");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_c.mtl_buffer(), &(0..third)).expect("map c");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_a.mtl_buffer(), &(0..third)).expect("unmap a");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_b.mtl_buffer(), &(0..third)).expect("unmap b");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_c.mtl_buffer(), &(0..third)).expect("unmap c");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 0);
}

#[test]
fn test_second_buffer_spans_existing_and_new_heap() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..pages_per_heap)).expect("map b spans heaps");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 2);
}

#[test]
fn test_unmap_spanning_buffer_keeps_heap_with_other_buffer() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = 2 * ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..pages_per_heap)).expect("map b spans heaps");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_b.mtl_buffer(), &(0..pages_per_heap)).expect("unmap b");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), 1);
}

#[test]
fn test_remap_two_buffers_after_unmap_does_not_grow_pool() {
    let ctx = create_context();
    let pages_per_heap = ctx.sparse_heap_pool().heap_capacity_pages();
    let half = pages_per_heap / 2;
    let cap = ctx.sparse_heap_pool().heap_capacity_bytes();
    let sparse_a = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);
    let sparse_b = common::helpers::sparse_buffer_create::<Metal>(&ctx, cap);

    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("map a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..half)).expect("map b");
    let initial = ctx.sparse_heap_pool().heaps_count();
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("unmap a");
    ctx.sparse_heap_pool_mut().unmap(&ctx, &sparse_b.mtl_buffer(), &(0..half)).expect("unmap b");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_a.mtl_buffer(), &(0..half)).expect("remap a");
    ctx.sparse_heap_pool_mut().map(&ctx, &sparse_b.mtl_buffer(), &(0..half)).expect("remap b");
    ctx.sparse_mappings_wait().expect("Failed to wait for mapping");

    assert_eq!(ctx.sparse_heap_pool().heaps_count(), initial);
}

#[test]
fn test_heap_capacity_pages_matches_byte_capacity() {
    let ctx = create_context();
    let pool = ctx.sparse_heap_pool();
    let page_size_bytes = pool.page_size().in_bytes();

    assert_eq!(pool.heap_capacity_pages(), pool.heap_capacity_bytes() / page_size_bytes);
}
