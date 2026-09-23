use std::mem::size_of;

use crate::backends::common::{Allocation, Backend, gpu_types::gemm::GemmAPrologueKind};

#[derive(Clone, Copy, PartialEq)]
pub enum Int8CodeLayout {
    Sequential,
    GroupedByNibble,
}

impl Int8CodeLayout {
    pub const fn for_right_bits(bits: u32) -> Option<Self> {
        match bits {
            4 => Some(Self::GroupedByNibble),
            8 => Some(Self::Sequential),
            _ => None,
        }
    }

    pub const fn is_grouped_by_nibble(self) -> bool {
        matches!(self, Self::GroupedByNibble)
    }

    pub const fn index(
        self,
        index: usize,
    ) -> usize {
        match self {
            Self::Sequential => index,
            Self::GroupedByNibble => {
                const NIBBLES_PER_BYTE: usize = 2;
                const CODES_PER_WORD: usize = size_of::<u32>() * NIBBLES_PER_BYTE;
                const NIBBLE_GROUP_SIZE: usize = CODES_PER_WORD / NIBBLES_PER_BYTE;
                let offset = index % CODES_PER_WORD;
                index - offset + (offset % NIBBLES_PER_BYTE) * NIBBLE_GROUP_SIZE + offset / NIBBLES_PER_BYTE
            },
        }
    }
}

pub enum MatmulA<'a, B: Backend> {
    FullPrecision {
        values: &'a Allocation<B>,
        offset: usize,
    },
    Int8Symmetric {
        values: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        group_sums: Option<&'a Allocation<B>>,
        scale_group_size: u32,
        code_layout: Int8CodeLayout,
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
