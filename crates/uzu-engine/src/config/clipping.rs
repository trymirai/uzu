use serde::{Deserialize, Serialize};

use crate::utils::strict_serde::DeserializeStrict;

type WireClippingBounds = Option<(Option<f32>, Option<f32>)>;

/// Optional inclusive clipping interval with one-sided bounds normalized to finite limits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "WireClippingBounds", into = "WireClippingBounds")]
pub struct ClippingBounds(Option<(f32, f32)>);

impl ClippingBounds {
    /// Finite bounds when clipping is configured.
    pub(crate) fn into_pair(self) -> Option<(f32, f32)> {
        self.0
    }
}

impl From<WireClippingBounds> for ClippingBounds {
    fn from(bounds: WireClippingBounds) -> Self {
        let Some((min, max)) = bounds else {
            return Self(None);
        };
        if min.is_none() && max.is_none() {
            return Self(None);
        }
        Self(Some((min.unwrap_or(f32::MIN), max.unwrap_or(f32::MAX))))
    }
}

impl From<ClippingBounds> for WireClippingBounds {
    fn from(bounds: ClippingBounds) -> Self {
        let Some((min, max)) = bounds.0 else {
            return None;
        };
        let min = (min != f32::MIN).then_some(min);
        let max = (max != f32::MAX).then_some(max);
        if min.is_none() && max.is_none() {
            return None;
        }

        Some((min, max))
    }
}

impl<'de> DeserializeStrict<'de> for ClippingBounds {}
