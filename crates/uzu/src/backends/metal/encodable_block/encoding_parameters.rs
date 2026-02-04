//! Encoding parameters for forward pass operations.

use crate::backends::metal::Metal;

pub type EncodingParameters<'a> =
    crate::encodable_block::EncodingParameters<'a, Metal>;
