use crate::backends::metal::{MTLSize, kernel::mlp::MlpActivationType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub threadgroup_rows: u32,
    pub threadgroup_cols: u32,
    pub threads_per_simdgroup_row: u32,
    pub threads_per_simdgroup_col: u32,
    pub elements_per_thread_row: u32,
    pub elements_per_thread_col: u32,
    pub activation: MlpActivationType,
}

impl PipelineConfiguration {
    pub fn output_elements_per_threadgroup(&self) -> u32 {
        self.threadgroup_rows
            * self.threads_per_simdgroup_row
            * self.elements_per_thread_row
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
    hidden_dim: i32,
    activation: MlpActivationType,
) -> PipelineConfiguration {
    let threadgroup_rows = if hidden_dim >= 4096 {
        8
    } else {
        4
    };

    PipelineConfiguration {
        threadgroup_rows,
        threadgroup_cols: 1,
        threads_per_simdgroup_row: 1,
        threads_per_simdgroup_col: 32,
        elements_per_thread_row: 4,
        elements_per_thread_col: 4,
        activation,
    }
}
