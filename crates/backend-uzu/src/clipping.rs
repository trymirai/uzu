use serde::{Deserialize, Serialize};

use crate::utils::strict_serde::DeserializeStrict;

type WireClippingBounds = Option<(Option<f32>, Option<f32>)>;

/// Inclusive clipping bounds; either side may be unbounded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "WireClippingBounds", into = "WireClippingBounds")]
pub struct ClippingBounds {
    pub min: Option<f32>,
    pub max: Option<f32>,
}

impl ClippingBounds {
    /// Both inclusive bounds.
    pub const fn bounded(
        min: f32,
        max: f32,
    ) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Inclusive lower bound.
    pub const fn lower_bounded(min: f32) -> Self {
        Self {
            min: Some(min),
            max: None,
        }
    }

    /// Inclusive upper bound.
    pub const fn upper_bounded(max: f32) -> Self {
        Self {
            min: None,
            max: Some(max),
        }
    }

    /// Clips a value to the configured bounds.
    pub fn apply(
        self,
        value: f32,
    ) -> f32 {
        let value = self.min.map_or(value, |min| value.max(min));
        self.max.map_or(value, |max| value.min(max))
    }
}

impl From<WireClippingBounds> for ClippingBounds {
    fn from(bounds: WireClippingBounds) -> Self {
        let (min, max) = bounds.unwrap_or_default();
        Self {
            min,
            max,
        }
    }
}

impl From<ClippingBounds> for WireClippingBounds {
    fn from(bounds: ClippingBounds) -> Self {
        if bounds.min.is_none() && bounds.max.is_none() {
            return None;
        }

        Some((bounds.min, bounds.max))
    }
}

impl<'de> DeserializeStrict<'de> for ClippingBounds {}
