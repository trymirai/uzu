use std::iter::successors;

use crate::{
    backends::common::{Backend, CommandBuffer},
    encodable_block::mixer::attention::rope::{RoPE, RoPEPositions},
};

#[derive(Clone, Copy)]
pub enum RoPECursor {
    Sequence {
        position: u32,
    },
    Grid {
        position: u32,
        width: u32,
        height: u32,
        row: u32,
    },
}

impl RoPECursor {
    pub fn begin_grid(
        self,
        width: u32,
        height: u32,
    ) -> Self {
        assert!(width > 0 && height > 0, "empty image grid");
        let Self::Sequence {
            position,
        } = self
        else {
            panic!("image grid begins inside another one");
        };

        Self::Grid {
            position,
            width,
            height,
            row: 0,
        }
    }

    pub fn in_grid(self) -> bool {
        matches!(self, Self::Grid { .. })
    }

    pub fn position(self) -> [u32; 3] {
        match self {
            Self::Sequence {
                position,
            } => [position; 3],
            Self::Grid {
                position,
                width,
                row,
                ..
            } => [position, position + row / width, position + row % width],
        }
    }

    pub fn advance(self) -> Self {
        match self {
            Self::Sequence {
                position,
            } => Self::Sequence {
                position: position + 1,
            },
            Self::Grid {
                position,
                width,
                height,
                row,
            } if row + 1 < width * height => Self::Grid {
                position,
                width,
                height,
                row: row + 1,
            },
            Self::Grid {
                position,
                width,
                height,
                ..
            } => Self::Sequence {
                position: position + width.max(height),
            },
        }
    }
}

impl<B: Backend> RoPE<B> {
    pub fn positions(
        &self,
        cursor: RoPECursor,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<RoPEPositions<B>, B::Error> {
        let positions = successors(Some(cursor), |cursor| Some(cursor.advance()))
            .take(row_count as usize)
            .map(RoPECursor::position)
            .collect::<Box<[_]>>();

        RoPEPositions::new(
            positions.len(),
            self.dim,
            |row, pair| match self.frequency_pairs_per_axis {
                Some(frequency_pairs_per_axis) => positions[row][interleaved_axis(pair, frequency_pairs_per_axis)],
                None => positions[row][0],
            },
            command_buffer,
        )
    }
}

/// Axis (0 — t, 1 — h, 2 — w) of frequency pair `pair` in the layout `[T H W T H W ... T T]`:
/// every third pair, while the axis has pairs left
fn interleaved_axis(
    pair: u32,
    frequency_pairs_per_axis: [u32; 3],
) -> usize {
    let [_, height_pairs, width_pairs] = frequency_pairs_per_axis;
    match pair % 3 {
        1 if pair < 3 * height_pairs => 1,
        2 if pair < 3 * width_pairs => 2,
        _ => 0,
    }
}
