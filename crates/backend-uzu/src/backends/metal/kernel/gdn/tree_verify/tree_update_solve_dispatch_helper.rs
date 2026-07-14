#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TreeUpdateSolveVariant {
    pub use_mxu: bool,
    pub bv: u32,
}

/// Fastest TreeUpdateSolve variant per device + shape, from the 9-chip fleet sweep
/// (raw table in .claude/plans/solve_u_kernel_optimization.md). The kernel is a
/// latency-bound serial recurrence at B=1 small-T, where the narrower BV16 wins on
/// occupancy; the tree size at which the wider BV32 takes over grows with GPU
/// width. MXU chips switch to the matrix unit past that same B=1 small-T corner.
pub(crate) fn tree_update_solve_variant(
    supports_mxu: bool,
    gpu_core_count: u32,
    batch_size: u32,
    tree_size: u32,
) -> TreeUpdateSolveVariant {
    if supports_mxu {
        let small_corner = batch_size == 1 && tree_size <= 64;
        return TreeUpdateSolveVariant {
            use_mxu: !small_corner,
            bv: if small_corner {
                16
            } else {
                32
            },
        };
    }
    let bv16 = if gpu_core_count >= 30 {
        batch_size == 1 || (batch_size <= 8 && tree_size <= 64)
    } else {
        let bv16_max_tree = if gpu_core_count >= 10 {
            128
        } else if gpu_core_count >= 8 {
            33
        } else {
            0
        };
        batch_size == 1 && tree_size <= bv16_max_tree
    };
    TreeUpdateSolveVariant {
        use_mxu: false,
        bv: if bv16 {
            16
        } else {
            32
        },
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/tree_update_solve_dispatch_test.rs"]
mod tests;
