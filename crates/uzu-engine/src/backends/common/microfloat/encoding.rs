use super::{MicrofloatError, MicrofloatFormat};

/// How packed microfloat bytes are interpreted, separate from matrix dimensions.
///
/// With MXFP4 group size 16, every 16 values along the input axis occupy eight
/// packed code bytes and share one E8M0 scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatEncoding {
    pub format: MicrofloatFormat,
    pub bits: u32,
    pub group_size: u32,
}

impl MicrofloatEncoding {
    pub fn new(
        format: MicrofloatFormat,
        bits: u32,
        group_size: u32,
    ) -> Result<Self, MicrofloatError> {
        if bits != 4 {
            return Err(MicrofloatError::UnsupportedBits {
                format,
                bits,
            });
        }
        if !matches!(group_size, 16 | 32) {
            return Err(MicrofloatError::UnsupportedGroupSize {
                format,
                group_size,
            });
        }
        Ok(Self {
            format,
            bits,
            group_size,
        })
    }
}
