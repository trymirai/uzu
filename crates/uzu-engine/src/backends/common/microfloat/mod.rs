mod decode;
mod encoding;
mod error;
mod format;
mod metadata;

pub use decode::{decode_e2m1, decode_e8m0, decode_mxfp4};
pub use encoding::MicrofloatEncoding;
pub use error::MicrofloatError;
pub use format::MicrofloatFormat;
pub use metadata::MicrofloatMetadata;

#[cfg(test)]
#[path = "../../../../unit/backends/common/microfloat/decode_test.rs"]
mod decode_test;
