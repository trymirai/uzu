use crate::backends::common::{Allocation, Backend};

pub enum MatmulA<'a, B: Backend> {
    FullPrecision {
        values: &'a Allocation<B>,
        offset: usize,
    },
    Int8Symmetric {
        values: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        group_size: u32,
    },
}

impl<'a, B: Backend> MatmulA<'a, B> {
    pub fn is_int8(&self) -> bool {
        !matches!(self, Self::FullPrecision { .. })
    }
}
