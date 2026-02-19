#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub transpose_matrix: bool,
    pub batch_pack: u32,
    pub threadgroup_rows: u32,
    pub threadgroup_cols: u32,
    pub threads_per_simdgroup_row: u32,
    pub threads_per_simdgroup_col: u32,
    pub elements_per_thread_row: u32,
    pub elements_per_thread_col: u32,
    pub non_contiguous_batch: bool,
    pub apply_output_scale_and_accumulate: bool,
}

const FORCE_TILESET_SMALL_BATCH: i32 = 0;

pub fn select_configuration(
    transpose_a: bool,
    transpose_b: bool,
    transpose_matrix: bool,
    batch_pack: u32,
    input_dimension: i32,
    output_dimension: i32,
    non_contiguous_batch: bool,
    apply_output_scale_and_accumulate: bool,
) -> PipelineConfiguration {
    let (threadgroup_rows, threadgroup_cols);
    let (threads_per_simdgroup_row, threads_per_simdgroup_col);
    let (elements_per_thread_row, elements_per_thread_col);

    if FORCE_TILESET_SMALL_BATCH == 1 && !transpose_matrix {
        threadgroup_rows = 4;
        threadgroup_cols = 1;
        threads_per_simdgroup_row = 1;
        threads_per_simdgroup_col = 32;
        elements_per_thread_row = 2;
        elements_per_thread_col = 4;
    } else if transpose_matrix {
        let mut simdgroup_thread_rows = 8;
        let mut simdgroup_thread_cols = 4;
        if input_dimension >= 8192 && output_dimension >= 2048 {
            simdgroup_thread_rows = 4;
            simdgroup_thread_cols = 8;
        }

        let threadgroup_simd_cols = if output_dimension >= 2048 {
            16
        } else if output_dimension >= 512 {
            4
        } else {
            2
        };

        let thread_output_cols = if output_dimension < 4 {
            1
        } else {
            4
        };

        threadgroup_rows = 1;
        threadgroup_cols = threadgroup_simd_cols;
        threads_per_simdgroup_row = simdgroup_thread_rows;
        threads_per_simdgroup_col = simdgroup_thread_cols;
        elements_per_thread_row = 4;
        elements_per_thread_col = thread_output_cols;
    } else {
        let mut threadgroup_simd_rows = if output_dimension >= 4096 {
            8
        } else {
            4
        };
        let mut simdgroup_thread_rows = 1;
        let mut simdgroup_thread_cols = 32;
        let mut threadgroup_simd_cols = 1;

        if input_dimension <= 64 {
            threadgroup_simd_rows = 1;
            simdgroup_thread_rows = 8;
            simdgroup_thread_cols = 4;
        } else if input_dimension >= 16 * output_dimension {
            threadgroup_simd_rows = 1;
            threadgroup_simd_cols = 8;
        }

        let thread_output_rows = if output_dimension < 4 {
            1
        } else {
            4
        };

        threadgroup_rows = threadgroup_simd_rows;
        threadgroup_cols = threadgroup_simd_cols;
        threads_per_simdgroup_row = simdgroup_thread_rows;
        threads_per_simdgroup_col = simdgroup_thread_cols;
        elements_per_thread_row = thread_output_rows;
        elements_per_thread_col = 4;
    }

    PipelineConfiguration {
        transpose_a,
        transpose_b,
        transpose_matrix,
        batch_pack,
        threadgroup_rows,
        threadgroup_cols,
        threads_per_simdgroup_row,
        threads_per_simdgroup_col,
        elements_per_thread_row,
        elements_per_thread_col,
        non_contiguous_batch,
        apply_output_scale_and_accumulate,
    }
}
