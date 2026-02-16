//! Embedding encodables (quantized and full-precision).

mod error;
mod quantized_readout;

pub use error::EmbeddingError;
pub use quantized_readout::QuantizedEmbeddingReadout;

use super::super::Metal;
use crate::encodable_block::{
    FullPrecisionEmbeddingLookup as GenericFullPrecisionEmbeddingLookup,
    FullPrecisionEmbeddingReadout as GenericFullPrecisionEmbeddingReadout,
    QuantizedEmbeddingLookup as GenericQuantizedEmbeddingLookup,
};

pub type FullPrecisionEmbeddingReadout = GenericFullPrecisionEmbeddingReadout<Metal>;
pub type FullPrecisionEmbeddingLookup = GenericFullPrecisionEmbeddingLookup<Metal>;
pub type QuantizedEmbeddingLookup = GenericQuantizedEmbeddingLookup<Metal>;
