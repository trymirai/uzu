use uzu_engine_macros::kernel;

use crate::backends::common::gpu_types::weaver::FrontierIdx;

pub const ABSENT: u32 = 7;

#[kernel(ConfidenceHeadTreeRelations)]
pub fn confidence_head_tree_relations(
    frontier: *const u32,
    slot_ancestors: *const u32,
    slot_rows: *const u32,
    relations: *mut u32,
    rows: u32,
    ancestor_stride: u32,
) {
    let (rows, stride) = (rows as usize, ancestor_stride as usize);
    let read = |buffer: *const u32, index: usize| unsafe { *buffer.add(index) } as usize;
    let tree_row = |row: usize| {
        if row == 0 {
            (0, usize::MAX)
        } else {
            (
                read(frontier, FrontierIdx::Depth as usize * (rows - 1) + row - 1),
                read(frontier, FrontierIdx::ParentSlot as usize * (rows - 1) + row - 1),
            )
        }
    };
    let ancestor_row = |(depth, parent): (usize, usize), level: usize| {
        read(
            slot_rows,
            if level + 1 == depth {
                parent
            } else {
                read(slot_ancestors, parent * stride + level)
            },
        )
    };
    for query in 0..rows {
        for key in 0..rows {
            let (query_row, key_row) = (tree_row(query), tree_row(key));
            let relation = if (query != 0 && query_row.0 == 0) || (key != 0 && key_row.0 == 0) {
                ABSENT
            } else if query == key {
                1
            } else if key_row.0 < query_row.0 && ancestor_row(query_row, key_row.0) == key {
                if key_row.0 + 1 == query_row.0 {
                    2
                } else {
                    4
                }
            } else if query_row.0 < key_row.0 && ancestor_row(key_row, query_row.0) == query {
                if query_row.0 + 1 == key_row.0 {
                    3
                } else {
                    5
                }
            } else if query_row.0 == key_row.0 && query_row.0 != 0 && query_row.1 == key_row.1 {
                6
            } else {
                0
            };
            unsafe { *relations.add(query * rows + key) = relation };
        }
    }
}
