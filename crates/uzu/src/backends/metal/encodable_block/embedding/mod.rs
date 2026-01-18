//! Embedding encodables (quantized and full-precision).

mod error;
mod full_precision_lookup;
mod full_precision_readout;
mod quantized_lookup;
mod quantized_readout;

pub use error::EmbeddingError;
pub use full_precision_lookup::FullPrecisionEmbeddingLookup;
pub use full_precision_readout::FullPrecisionEmbeddingReadout;
pub use quantized_lookup::QuantizedEmbeddingLookup;
pub use quantized_readout::QuantizedEmbeddingReadout;
