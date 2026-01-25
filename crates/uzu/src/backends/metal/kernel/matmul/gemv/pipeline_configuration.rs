use crate::backends::metal::MTLSize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub transpose_matrix: bool,
    pub threadgroup_rows: u32,
    pub threadgroup_cols: u32,
    pub threads_per_simdgroup_row: u32,
    pub threads_per_simdgroup_col: u32,
    pub elements_per_thread_row: u32,
    pub elements_per_thread_col: u32,
    pub non_contiguous_batch: bool,
    pub do_axpby: bool,
}

impl PipelineConfiguration {
    pub fn output_elements_per_threadgroup(&self) -> u32 {
        if self.transpose_matrix {
            self.threadgroup_cols
                * self.threads_per_simdgroup_col
                * self.elements_per_thread_col
        } else {
            self.threadgroup_rows
                * self.threads_per_simdgroup_row
                * self.elements_per_thread_row
        }
    }

    pub fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize::new(
            32,
            self.threadgroup_cols as usize,
            self.threadgroup_rows as usize,
        )
    }
}

pub fn select_configuration(
    transpose_a: bool,
    transpose_b: bool,
    transpose_matrix: bool,
    input_dimension: i32,
    output_dimension: i32,
    non_contiguous_batch: bool,
    do_axpby: bool,
) -> PipelineConfiguration {
    let (threadgroup_rows, threadgroup_cols);
    let (threads_per_simdgroup_row, threads_per_simdgroup_col);
    let (elements_per_thread_row, elements_per_thread_col);

    if transpose_matrix {
        let mut sm = 8;
        let mut sn = 4;
        if input_dimension >= 8192 && output_dimension >= 2048 {
            sm = 4;
            sn = 8;
        }

        let bn = if output_dimension >= 2048 {
            16
        } else if output_dimension >= 512 {
            4
        } else {
            2
        };

        let tn = if output_dimension < 4 {
            1
        } else {
            4
        };

        threadgroup_rows = 1;
        threadgroup_cols = bn;
        threads_per_simdgroup_row = sm;
        threads_per_simdgroup_col = sn;
        elements_per_thread_row = 4;
        elements_per_thread_col = tn;
    } else {
        let mut bm = if output_dimension >= 4096 {
            8
        } else {
            4
        };
        let mut sm = 1;
        let mut sn = 32;
        let mut bn = 1;

        if input_dimension <= 64 {
            bm = 1;
            sm = 8;
            sn = 4;
        } else if input_dimension >= 16 * output_dimension {
            bm = 1;
            bn = 8;
        }

        let tm = if output_dimension < 4 {
            1
        } else {
            4
        };

        threadgroup_rows = bm;
        threadgroup_cols = bn;
        threads_per_simdgroup_row = sm;
        threads_per_simdgroup_col = sn;
        elements_per_thread_row = tm;
        elements_per_thread_col = 4;
    }

    PipelineConfiguration {
        transpose_a,
        transpose_b,
        transpose_matrix,
        threadgroup_rows,
        threadgroup_cols,
        threads_per_simdgroup_row,
        threads_per_simdgroup_col,
        elements_per_thread_row,
        elements_per_thread_col,
        non_contiguous_batch,
        do_axpby,
    }
}
