//! Unified normalization encodables: LayerNorm, RMSNorm, QKNorm.

mod qk_norm;

pub use qk_norm::QKNorm;

use super::Metal;
use crate::encodable_block::{
    LayerNorm as GenericLayerNorm, Normalization as GenericNormalization, RMSNorm as GenericRMSNorm,
};

pub type LayerNorm = GenericLayerNorm<Metal>;
pub type RMSNorm = GenericRMSNorm<Metal>;
pub type Normalization = GenericNormalization<Metal>;
