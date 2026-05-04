use crate::backends::{common::Backend, metal::Metal};

type DenseBuffer = <Metal as Backend>::DenseBuffer;

pub(crate) enum GemmWeightsBuffers<'a> {
    FullPrecision {
        weights: &'a DenseBuffer,
    },
    Mlx {
        weights: &'a DenseBuffer,
        scales: &'a DenseBuffer,
        biases: &'a DenseBuffer,
    },
    Awq {
        weights: &'a DenseBuffer,
        scales: &'a DenseBuffer,
        zero_points: &'a DenseBuffer,
    },
}
