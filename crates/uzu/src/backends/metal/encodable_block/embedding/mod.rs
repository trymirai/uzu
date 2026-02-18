//! Embedding encodables (quantized and full-precision).

use super::super::Metal;
use crate::encodable_block::{
    FullPrecisionEmbeddingLookup as GenericFullPrecisionEmbeddingLookup,
    FullPrecisionEmbeddingReadout as GenericFullPrecisionEmbeddingReadout,
    QuantizedEmbeddingLookup as GenericQuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout as GenericQuantizedEmbeddingReadout,
    QuantizedEmbeddingReadoutError as GenericQuantizedEmbeddingReadoutError,
};

pub type EmbeddingError = GenericQuantizedEmbeddingReadoutError<Metal>;
pub type QuantizedEmbeddingReadout = GenericQuantizedEmbeddingReadout<Metal>;
pub type FullPrecisionEmbeddingReadout = GenericFullPrecisionEmbeddingReadout<Metal>;
pub type FullPrecisionEmbeddingLookup = GenericFullPrecisionEmbeddingLookup<Metal>;
pub type QuantizedEmbeddingLookup = GenericQuantizedEmbeddingLookup<Metal>;
