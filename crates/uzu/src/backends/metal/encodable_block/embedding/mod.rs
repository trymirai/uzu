//! Embedding encodables (quantized and full-precision).

mod error;
mod full_precision_readout;
mod quantized_lookup;
mod quantized_readout;

use super::super::Metal;
use crate::encodable_block::FullPrecisionEmbeddingLookup as GenericFullPrecisionEmbeddingLookup;

pub use error::EmbeddingError;
pub use full_precision_readout::FullPrecisionEmbeddingReadout;
pub use quantized_lookup::QuantizedEmbeddingLookup;
pub use quantized_readout::QuantizedEmbeddingReadout;

pub type FullPrecisionEmbeddingLookup = GenericFullPrecisionEmbeddingLookup<Metal>;
