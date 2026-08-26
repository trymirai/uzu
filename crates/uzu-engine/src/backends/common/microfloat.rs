use thiserror::Error;
use uzu_engine_macros::uzu_config;

#[derive(Copy, Eq)]
#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatFormat {
    Mxfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MicrofloatError {
    #[error("unsupported {format:?} bit width: {bits}")]
    UnsupportedBits {
        format: MicrofloatFormat,
        bits: u32,
    },
    #[error("unsupported {format:?} group size: {group_size}")]
    UnsupportedGroupSize {
        format: MicrofloatFormat,
        group_size: u32,
    },
    #[error("microfloat rows and columns must be nonzero")]
    EmptyShape,
    #[error("microfloat columns {columns} are not divisible by group size {group_size}")]
    MisalignedColumns {
        columns: u32,
        group_size: u32,
    },
    #[error("microfloat storage size overflows usize")]
    SizeOverflow,
}

/// How packed microfloat bytes are interpreted, separate from matrix dimensions.
///
/// With MXFP4 group size 16, every 16 values along the input axis occupy eight
/// packed code bytes and share one E8M0 scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatEncoding {
    format: MicrofloatFormat,
    group_size: u32,
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
            group_size,
        })
    }

    pub fn format(self) -> MicrofloatFormat {
        self.format
    }

    pub fn bits(self) -> u32 {
        4
    }

    pub fn group_size(self) -> u32 {
        self.group_size
    }
}

/// Physical shape and derived strides for one microfloat matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatMetadata {
    encoding: MicrofloatEncoding,
    rows: u32,
    columns: u32,
}

impl MicrofloatMetadata {
    pub fn new(
        encoding: MicrofloatEncoding,
        rows: u32,
        columns: u32,
    ) -> Result<Self, MicrofloatError> {
        if rows == 0 || columns == 0 {
            return Err(MicrofloatError::EmptyShape);
        }
        let group_size = encoding.group_size();
        if !columns.is_multiple_of(group_size) {
            return Err(MicrofloatError::MisalignedColumns {
                columns,
                group_size,
            });
        }
        let metadata = Self {
            encoding,
            rows,
            columns,
        };
        metadata.checked_code_matrix_stride().ok_or(MicrofloatError::SizeOverflow)?;
        metadata.checked_scale_matrix_stride().ok_or(MicrofloatError::SizeOverflow)?;
        Ok(metadata)
    }

    pub fn format(self) -> MicrofloatFormat {
        self.encoding.format()
    }

    pub fn bits(self) -> u32 {
        self.encoding.bits()
    }

    pub fn group_size(self) -> u32 {
        self.encoding.group_size()
    }

    pub fn rows(self) -> u32 {
        self.rows
    }

    pub fn columns(self) -> u32 {
        self.columns
    }

    pub fn code_row_stride(self) -> usize {
        self.columns as usize / 2
    }

    pub fn scale_row_stride(self) -> usize {
        self.columns as usize / self.group_size() as usize
    }

    pub fn code_matrix_stride(self) -> usize {
        self.checked_code_matrix_stride().expect("MicrofloatMetadata validates code storage size")
    }

    pub fn scale_matrix_stride(self) -> usize {
        self.checked_scale_matrix_stride().expect("MicrofloatMetadata validates scale storage size")
    }

    pub fn required_code_bytes(self) -> usize {
        self.code_matrix_stride()
    }

    pub fn required_scale_bytes(self) -> usize {
        self.scale_matrix_stride()
    }

    fn checked_code_matrix_stride(self) -> Option<usize> {
        (self.rows as usize).checked_mul(self.code_row_stride())
    }

    fn checked_scale_matrix_stride(self) -> Option<usize> {
        (self.rows as usize).checked_mul(self.scale_row_stride())
    }
}

#[inline]
pub fn decode_e2m1(code: u8) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(code & 0x0f)]
}

#[inline]
pub fn decode_e8m0(exponent: u8) -> f32 {
    match exponent {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        exponent => f32::from_bits(u32::from(exponent) << 23),
    }
}

#[inline]
pub fn decode_mxfp4(
    code: u8,
    exponent: u8,
    outer_scale: f32,
) -> f32 {
    decode_e2m1(code) * decode_e8m0(exponent) * outer_scale
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn decodes_e2m1_and_e8m0_edges() {
        let expected = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, value) in expected.into_iter().enumerate() {
            assert_eq!(decode_e2m1(code as u8), value);
            assert_eq!(decode_e2m1(code as u8 | 0b1000), -value);
        }
        assert_eq!(decode_e8m0(0).to_bits(), 0x0040_0000);
        assert_eq!(decode_e8m0(127), 1.0);
        assert!(decode_e8m0(255).is_nan());
    }

    #[uzu_test]
    fn validates_supported_mxfp4_encodings() {
        for group_size in [16, 32] {
            let encoding =
                MicrofloatEncoding::new(MicrofloatFormat::Mxfp4, 4, group_size).expect("supported MXFP4 encoding");
            let metadata = MicrofloatMetadata::new(encoding, 3, 32).expect("supported MXFP4 metadata");
            assert_eq!(metadata.required_code_bytes(), 48);
            assert_eq!(metadata.required_scale_bytes(), 3 * 32 / group_size as usize);
        }
        assert!(matches!(
            MicrofloatEncoding::new(MicrofloatFormat::Mxfp4, 8, 16),
            Err(MicrofloatError::UnsupportedBits {
                bits: 8,
                ..
            })
        ));
    }
}
