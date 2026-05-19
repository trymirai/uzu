#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemvSpecialization {
    pub threadgroup_rows: u32,
    pub threadgroup_cols: u32,
    pub threads_per_simdgroup_row: u32,
    pub threads_per_simdgroup_col: u32,
    pub elements_per_thread_row: u32,
    pub elements_per_thread_col: u32,
    pub is_accumulate: bool,
    pub is_bias: bool,
}

impl GemvSpecialization {
    pub fn precompile_configs(data_type: crate::DataType) -> &'static [Self] {
        use crate::DataType;
        match data_type {
            DataType::BF16 => &[
                Self {
                    threadgroup_rows: 4,
                    threadgroup_cols: 1,
                    threads_per_simdgroup_row: 1,
                    threads_per_simdgroup_col: 32,
                    elements_per_thread_row: 4,
                    elements_per_thread_col: 4,
                    is_accumulate: false,
                    is_bias: false,
                },
                Self {
                    threadgroup_rows: 4,
                    threadgroup_cols: 1,
                    threads_per_simdgroup_row: 1,
                    threads_per_simdgroup_col: 32,
                    elements_per_thread_row: 4,
                    elements_per_thread_col: 4,
                    is_accumulate: false,
                    is_bias: true,
                },
                Self {
                    threadgroup_rows: 8,
                    threadgroup_cols: 1,
                    threads_per_simdgroup_row: 1,
                    threads_per_simdgroup_col: 32,
                    elements_per_thread_row: 4,
                    elements_per_thread_col: 4,
                    is_accumulate: false,
                    is_bias: false,
                },
                Self {
                    threadgroup_rows: 8,
                    threadgroup_cols: 1,
                    threads_per_simdgroup_row: 1,
                    threads_per_simdgroup_col: 32,
                    elements_per_thread_row: 4,
                    elements_per_thread_col: 4,
                    is_accumulate: false,
                    is_bias: true,
                },
            ],
            DataType::F16 => &[Self {
                threadgroup_rows: 8,
                threadgroup_cols: 1,
                threads_per_simdgroup_row: 1,
                threads_per_simdgroup_col: 32,
                elements_per_thread_row: 4,
                elements_per_thread_col: 4,
                is_accumulate: false,
                is_bias: false,
            }],
            DataType::F32 => &[Self {
                threadgroup_rows: 8,
                threadgroup_cols: 1,
                threads_per_simdgroup_row: 1,
                threads_per_simdgroup_col: 32,
                elements_per_thread_row: 4,
                elements_per_thread_col: 4,
                is_accumulate: false,
                is_bias: false,
            }],
            _ => &[],
        }
    }

    pub fn output_rows_per_threadgroup(&self) -> u32 {
        self.threadgroup_rows * self.threads_per_simdgroup_row * self.elements_per_thread_row
    }

    pub fn select(
        input_dimension: u32,
        output_dimension: u32,
        is_accumulate: bool,
        is_bias: bool,
    ) -> Self {
        let (threadgroup_rows, threadgroup_cols);
        let (threads_per_simdgroup_row, threads_per_simdgroup_col);
        let (elements_per_thread_row, elements_per_thread_col);

        let threadgroup_simd_rows;
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
        } else if output_dimension >= 4096 {
            threadgroup_simd_rows = 8;
        } else {
            threadgroup_simd_rows = 4;
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

        Self {
            threadgroup_rows,
            threadgroup_cols,
            threads_per_simdgroup_row,
            threads_per_simdgroup_col,
            elements_per_thread_row,
            elements_per_thread_col,
            is_accumulate,
            is_bias,
        }
    }
}
