use half::bf16;
use uzu_engine_macros::kernel;

use super::head::NONE;
use crate::backends::common::gpu_types::weaver::FrontierIdx;

#[kernel(ConfidenceHeadTreeInput)]
pub fn confidence_head_tree_input(
    depth_embedding: *const bf16,
    child_in: *const bf16,
    node_in: *const bf16,
    frontier: *const u32,
    row_slots: *const u32,
    output: *mut bf16,
    rows: u32,
    width: u32,
) {
    let (rows, width) = (rows as usize, width as usize);
    let value = |buffer: *const bf16, index: usize| unsafe { (*buffer.add(index)).to_f32() };
    for row in 0..rows {
        let depth =
            if row == 0 { 0 } else { unsafe { *frontier.add(FrontierIdx::Depth as usize * (rows - 1) + row - 1) } };
        let slot = unsafe { *row_slots.add(row) };
        for unit in 0..width {
            let mut sum = 0.0;
            if row == 0 || depth != 0 {
                sum = value(depth_embedding, depth as usize * width + unit);
                if row != 0 {
                    sum += value(child_in, (row - 1) * width + unit);
                }
                if slot != NONE {
                    sum += value(node_in, slot as usize * width + unit);
                }
            }
            unsafe { *output.add(row * width + unit) = bf16::from_f32(sum) };
        }
    }
}
