use crate::backends::common::{Allocation, Backend, gpu_types::gemm::GemmAPrologueKind};

pub enum MatmulA<'a, B: Backend> {
    FullPrecision {
        values: &'a Allocation<B>,
        offset: usize,
    },
    Int8Symmetric {
        values: &'a Allocation<B>,
        scales: &'a Allocation<B>,
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
