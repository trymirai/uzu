//! Unified normalization encodables: LayerNorm, RMSNorm, QKNorm.

use super::Metal;
use crate::encodable_block::{
    LayerNorm as GenericLayerNorm, Normalization as GenericNormalization, QKNorm as GenericQKNorm,
    RMSNorm as GenericRMSNorm,
};

pub type LayerNorm = GenericLayerNorm<Metal>;
pub type RMSNorm = GenericRMSNorm<Metal>;
pub type Normalization = GenericNormalization<Metal>;
pub type QKNorm = GenericQKNorm<Metal>;
