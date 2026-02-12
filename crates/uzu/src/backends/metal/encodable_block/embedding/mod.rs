//! Embedding encodables (quantized and full-precision).

mod error;
mod full_precision_readout;
mod quantized_readout;

pub use error::EmbeddingError;
pub use full_precision_readout::FullPrecisionEmbeddingReadout;
pub use quantized_readout::QuantizedEmbeddingReadout;

use super::super::Metal;
use crate::encodable_block::{
    FullPrecisionEmbeddingLookup as GenericFullPrecisionEmbeddingLookup,
    QuantizedEmbeddingLookup as GenericQuantizedEmbeddingLookup,
};

pub type FullPrecisionEmbeddingLookup = GenericFullPrecisionEmbeddingLookup<Metal>;
pub type QuantizedEmbeddingLookup = GenericQuantizedEmbeddingLookup<Metal>;
