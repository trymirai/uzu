use num_traits::Float;

use crate::{
    backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
    encodable_block::normalization::{PostLayerScalar, ShortcutMode},
};

/// One raw Normalization dispatch shared by the CPU and Vulkan runners: payloads, dimensions, scalars, and the model
/// layer's shortcut and post-layer modes, from which both runners derive the same ten specializations.
#[derive(Clone)]
pub struct NormalizationCase<I, A> {
    pub input: Vec<I>,
    pub scales: Option<Vec<A>>,
    pub biases: Option<Vec<A>>,
    /// Initial shortcut, bound unless `shortcut_mode` is `ShortcutMode::None`.
    pub shortcut: Vec<I>,
    pub hadamard_factors: Option<Vec<i32>>,
    pub batch_size: u32,
    pub element_count: u32,
    pub epsilon: f32,
    pub scale_offset: f32,
    pub in_place: bool,
    pub subtract_mean: bool,
    pub full_layer: bool,
    pub shortcut_mode: ShortcutMode,
    pub post_layer_scalar: PostLayerScalar,
}

impl<I: Float, A: Float> NormalizationCase<I, A> {
    /// Plain RMS normalization of `batch_size` rows of `element_count` values in [-4, 4]; `seed` varies the pattern.
    pub fn new(
        batch_size: u32,
        element_count: u32,
        seed: usize,
    ) -> Self {
        let length = batch_size as usize * element_count as usize;
        let pattern = |seed: usize, range: f32| {
            (0..length).map(|i| I::from(((i * 37 + seed) % 199) as f32 / 99.0 * range - range).unwrap()).collect()
        };
        Self {
            input: pattern(seed, 4.0),
            scales: None,
            biases: None,
            shortcut: pattern(seed + 11, 2.0),
            hadamard_factors: None,
            batch_size,
            element_count,
            epsilon: 1e-5,
            scale_offset: 0.0,
            in_place: false,
            subtract_mean: false,
            full_layer: true,
            shortcut_mode: ShortcutMode::None,
            post_layer_scalar: PostLayerScalar::None,
        }
    }

    pub fn subtract_mean(mut self) -> Self {
        self.subtract_mean = true;
        self
    }

    /// Scales in [0.5, 1.5] plus `scale_offset`, applied in FP32 when `full_layer` and in the output type otherwise.
    pub fn scales(
        mut self,
        full_layer: bool,
        scale_offset: f32,
    ) -> Self {
        let count = self.element_count as usize;
        self.scales = Some((0..count).map(|i| A::from(0.5 + ((i * 13) % 17) as f32 / 16.0).unwrap()).collect());
        self.full_layer = full_layer;
        self.scale_offset = scale_offset;
        self
    }

    /// Biases in [-0.5, 0.5].
    pub fn biases(mut self) -> Self {
        let count = self.element_count as usize;
        self.biases = Some((0..count).map(|i| A::from(((i * 7) % 11) as f32 / 10.0 - 0.5).unwrap()).collect());
        self
    }

    pub fn shortcut(
        mut self,
        mode: ShortcutMode,
    ) -> Self {
        self.shortcut_mode = mode;
        self
    }

    pub fn post(
        mut self,
        post_layer_scalar: PostLayerScalar,
    ) -> Self {
        self.post_layer_scalar = post_layer_scalar;
        self
    }

    /// Input random Hadamard transform with ±1 factors; `element_count` must be a multiple of the block size.
    pub fn hadamard(mut self) -> Self {
        assert!(self.element_count.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE), "RHT needs whole blocks");
        let count = self.element_count as usize;
        self.hadamard_factors = Some(
            (0..count)
                .map(|i| {
                    if (i * 5) % 3 == 0 {
                        -1
                    } else {
                        1
                    }
                })
                .collect(),
        );
        self
    }

    pub fn in_place(mut self) -> Self {
        self.in_place = true;
        self
    }

    /// The ten specializations in kernel signature order. The shortcut and post-layer flags map from the modes exactly
    /// as the model layer maps them, so residual_add implies copy_to_shortcut.
    pub fn specializations(&self) -> [bool; 10] {
        let (copy_to_shortcut, residual_add) = match self.shortcut_mode {
            ShortcutMode::None => (false, false),
            ShortcutMode::Copy => (true, false),
            ShortcutMode::Add => (true, true),
        };
        let (scale_residual_sum, scale_output) = match self.post_layer_scalar {
            PostLayerScalar::None => (false, false),
            PostLayerScalar::ScaleResidualSum(_) => (true, false),
            PostLayerScalar::ScaleOutput(_) => (false, true),
        };
        [
            self.in_place,
            self.subtract_mean,
            self.full_layer,
            copy_to_shortcut,
            residual_add,
            self.hadamard_factors.is_some(),
            scale_residual_sum,
            scale_output,
            self.biases.is_some(),
            self.scales.is_some(),
        ]
    }

    pub fn post_layer_scalar_value(&self) -> f32 {
        match self.post_layer_scalar {
            PostLayerScalar::None => 1.0,
            PostLayerScalar::ScaleResidualSum(value) | PostLayerScalar::ScaleOutput(value) => value,
        }
    }

    /// The output before the dispatch: the input for in-place cases (whose input and output types are equal),
    /// otherwise `sentinel`.
    pub fn initial_output<O: Float>(
        &self,
        sentinel: O,
    ) -> Vec<O> {
        match self.in_place {
            true => self.input.iter().map(|&value| O::from(value).unwrap()).collect(),
            false => vec![sentinel; self.input.len()],
        }
    }

    /// The fusion path, without dimensions.
    pub fn path(&self) -> String {
        let affine = match (&self.scales, &self.biases) {
            (Some(_), _) if self.full_layer => "scales(full_layer)",
            (Some(_), _) => "scales(only_normalization)",
            (None, _) => "no scales",
        };
        format!(
            "{} {affine}{} shortcut {:?} post {:?}{}{}",
            if self.subtract_mean {
                "layer_norm"
            } else {
                "rms"
            },
            if self.biases.is_some() {
                " biases"
            } else {
                ""
            },
            self.shortcut_mode,
            self.post_layer_scalar,
            if self.hadamard_factors.is_some() {
                " rht"
            } else {
                ""
            },
            if self.in_place {
                " in_place"
            } else {
                ""
            },
        )
    }
}
