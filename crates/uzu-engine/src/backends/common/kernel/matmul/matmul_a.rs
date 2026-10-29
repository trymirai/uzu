use std::mem::size_of;

use crate::backends::common::kernel::ActivationQuantization;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8CodeLayout {
    Sequential,
    GroupedByNibble,
}

impl Int8CodeLayout {
    pub const fn from_grouped_by_nibble(grouped: bool) -> Self {
        match grouped {
            true => Self::GroupedByNibble,
            false => Self::Sequential,
        }
    }

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
                // [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7]
                const NIBBLES_PER_BYTE: usize = 2;
                const CODES_PER_WORD: usize = size_of::<u32>() * NIBBLES_PER_BYTE;
                const NIBBLE_GROUP_SIZE: usize = CODES_PER_WORD / NIBBLES_PER_BYTE;
                let offset = index % CODES_PER_WORD;
                index - offset + (offset % NIBBLES_PER_BYTE) * NIBBLE_GROUP_SIZE + offset / NIBBLES_PER_BYTE
            },
        }
    }
}

#[derive(Clone, Copy)]
pub enum MatmulA<T> {
    FullPrecision {
        values: T,
        offset: usize,
    },
    PackedHalf {
        values: T,
        group_size: u32,
    },
    Int8Symmetric {
        values: T,
        scales: T,
        group_sums: Option<T>,
        quantization: ActivationQuantization,
    },
    Trellis {
        values: T,
        column_group_sums: T,
        scales: T,
    },
}

impl<T> MatmulA<T> {
    pub fn full_precision(values: T) -> Self {
        Self::FullPrecision {
            values,
            offset: 0,
        }
    }

    pub fn as_ref(&self) -> MatmulA<&T> {
        match self {
            Self::FullPrecision {
                values,
                offset,
            } => MatmulA::FullPrecision {
                values,
                offset: *offset,
            },
            Self::PackedHalf {
                values,
                group_size,
            } => MatmulA::PackedHalf {
                values,
                group_size: *group_size,
            },
            Self::Int8Symmetric {
                values,
                scales,
                group_sums,
                quantization,
            } => MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums: group_sums.as_ref(),
                quantization: *quantization,
            },
            Self::Trellis {
                values,
                column_group_sums,
                scales,
            } => MatmulA::Trellis {
                values,
                column_group_sums,
                scales,
            },
        }
    }
}

/// Matches `PackedHalfLayout` in `packed_half_layout.h`.
pub fn packed_half_bytes(
    k: u32,
    group_size: u32,
) -> usize {
    assert!(matches!(group_size, 32 | 64) && k.is_multiple_of(group_size));
    (k / group_size) as usize * (32 * (group_size as usize / 8) + 16) * size_of::<u32>()
}
