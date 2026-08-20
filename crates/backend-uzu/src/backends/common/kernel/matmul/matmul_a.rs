use crate::backends::common::{Allocation, Backend, gpu_types::gemm::GemmAPrologueKind};

pub enum MatmulA<'a, B: Backend> {
    FullPrecision {
        values: &'a Allocation<B>,
        /// Byte offset from the start of `values`.
        offset: usize,
    },
    Int8Symmetric {
        values: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        group_sums: Option<&'a Allocation<B>>,
        group_size: u32,
    },
}

impl<'a, B: Backend> MatmulA<'a, B> {
    pub fn prologue_kind(&self) -> GemmAPrologueKind {
        match self {
            Self::FullPrecision {
                ..
            } => GemmAPrologueKind::FullPrecision,
            Self::Int8Symmetric {
                ..
            } => GemmAPrologueKind::Int8Symmetric,
        }
    }
}
