use half::bf16;

use super::patch_coordinates;
use crate::{
    backends::Backend,
    data_type::DataTypeRepr,
    safetensors::{ParameterLoaderError, ParameterTree},
};

/// Learned absolute positions on a `side x side` grid, bilinearly resampled to each image's patch grid on the CPU.
pub struct InterpolatedPositionEmbedding<T> {
    /// `[side * side, model_dim]`, in the model's data type.
    table: Box<[T]>,
    side: usize,
    model_dim: usize,
}

/// The position embedding in the vision model's data type.
pub enum PositionEmbedding {
    BF16(InterpolatedPositionEmbedding<bf16>),
    F32(InterpolatedPositionEmbedding<f32>),
}

impl<T: DataTypeRepr> InterpolatedPositionEmbedding<T> {
    pub async fn load<B: Backend>(
        num_positions: u32,
        model_dim: u32,
        tree: &ParameterTree<'_, B>,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let side = num_positions.isqrt();
        assert_eq!(side * side, num_positions, "position table size {num_positions} isn't a square");
        Ok(Self {
            table: tree.read_slice::<T>("table", &[num_positions, model_dim]).await?,
            side: side as usize,
            model_dim: model_dim as usize,
        })
    }

    /// `[height * width, model_dim]` embeddings in HF's patch order. Corners are aligned like HF's
    /// `linspace(0, side - 1, n)`; the blend is in f32 like HF, and each value is rounded to `T`.
    pub fn interpolate(
        &self,
        height: u32,
        width: u32,
        merge: u32,
    ) -> Vec<T> {
        let side = self.side;
        // For each output row (or column): the two table rows to blend and the weight of the second one.
        let axis = |count: u32| -> Vec<(usize, usize, f32)> {
            let step = (side - 1) as f64 / count.saturating_sub(1).max(1) as f64;
            (0..count)
                .map(|index| {
                    // numpy's float32 linspace: f64 positions with an exact endpoint, then rounded to f32.
                    let position = if count > 1 && index == count - 1 {
                        (side - 1) as f32
                    } else {
                        (index as f64 * step) as f32
                    };
                    let lower = position.floor();
                    (lower as usize, (lower as usize + 1).min(side - 1), position - lower)
                })
                .collect()
        };
        let (rows, cols) = (axis(height), axis(width));

        let mut output = Vec::with_capacity((height * width) as usize * self.model_dim);
        for (row, col) in patch_coordinates(height, width, merge) {
            let ((row0, row1, row_weight), (col0, col1, col_weight)) = (rows[row as usize], cols[col as usize]);
            let corners = [
                (row0, col0, (1.0 - row_weight) * (1.0 - col_weight)),
                (row0, col1, (1.0 - row_weight) * col_weight),
                (row1, col0, row_weight * (1.0 - col_weight)),
                (row1, col1, row_weight * col_weight),
            ];
            output.extend((0..self.model_dim).map(|channel| {
                let blend = corners
                    .iter()
                    .map(|&(row, col, weight)| {
                        weight * self.table[(row * side + col) * self.model_dim + channel].to_f32().unwrap()
                    })
                    .sum::<f32>();
                T::from(blend).unwrap()
            }));
        }
        output
    }
}
