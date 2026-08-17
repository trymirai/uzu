use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrofloatFormat {
    Mxfp4,
    Nvfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrofloatLayout {
    OutputInput,
    InputOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MicrofloatError {
    #[error("unsupported microfloat format: {0:?}")]
    UnsupportedFormat(MicrofloatFormat),
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
    #[error("unsupported microfloat matrix layout: {0:?}")]
    UnsupportedLayout(MicrofloatLayout),
    #[error("microfloat matrix count, rows, and columns must be nonzero")]
    EmptyShape,
    #[error("microfloat columns {columns} are not divisible by group size {group_size}")]
    MisalignedColumns {
        columns: u32,
        group_size: u32,
    },
    #[error("microfloat storage size overflows usize")]
    SizeOverflow,
}

/// Packed E2M1 values with per-group E8M0 scales.
///
/// Group 32 is the OCP MXFP4 block size. Group 16 is the converter-defined
/// GPT-OSS layout, which retains the same E2M1/E8M0 encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatMetadata {
    format: MicrofloatFormat,
    group_size: u32,
    layout: MicrofloatLayout,
    matrix_count: u32,
    rows: u32,
    columns: u32,
}

impl MicrofloatMetadata {
    pub fn new(
        format: MicrofloatFormat,
        bits: u32,
        group_size: u32,
        layout: MicrofloatLayout,
        matrix_count: u32,
        rows: u32,
        columns: u32,
    ) -> Result<Self, MicrofloatError> {
        if format != MicrofloatFormat::Mxfp4 {
            return Err(MicrofloatError::UnsupportedFormat(format));
        }
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
        if layout != MicrofloatLayout::OutputInput {
            return Err(MicrofloatError::UnsupportedLayout(layout));
        }
        if matrix_count == 0 || rows == 0 || columns == 0 {
            return Err(MicrofloatError::EmptyShape);
        }
        if !columns.is_multiple_of(group_size) {
            return Err(MicrofloatError::MisalignedColumns {
                columns,
                group_size,
            });
        }
        let metadata = Self {
            format,
            group_size,
            layout,
            matrix_count,
            rows,
            columns,
        };
        metadata.code_matrix_stride().checked_mul(matrix_count as usize).ok_or(MicrofloatError::SizeOverflow)?;
        metadata.scale_matrix_stride().checked_mul(matrix_count as usize).ok_or(MicrofloatError::SizeOverflow)?;
        Ok(metadata)
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

    pub fn layout(self) -> MicrofloatLayout {
        self.layout
    }

    pub fn matrix_count(self) -> u32 {
        self.matrix_count
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
        self.columns as usize / self.group_size as usize
    }

    pub fn code_matrix_stride(self) -> usize {
        self.rows as usize * self.code_row_stride()
    }

    pub fn scale_matrix_stride(self) -> usize {
        self.rows as usize * self.scale_row_stride()
    }

    pub fn required_code_bytes(self) -> usize {
        self.matrix_count as usize * self.code_matrix_stride()
    }

    pub fn required_scale_bytes(self) -> usize {
        self.matrix_count as usize * self.scale_matrix_stride()
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
    global_scale: f32,
) -> f32 {
    decode_e2m1(code) * decode_e8m0(exponent) * global_scale
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn decodes_e2m1_and_e8m0_edges() {
        let expected = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, value) in expected.into_iter().enumerate() {
            assert_eq!(decode_e2m1(code as u8), value);
            assert_eq!(decode_e2m1(code as u8 | 8), -value);
        }
        assert_eq!(decode_e8m0(0).to_bits(), 0x0040_0000);
        assert_eq!(decode_e8m0(127), 1.0);
        assert!(decode_e8m0(255).is_nan());
    }

    #[uzu_test]
    fn validates_supported_mxfp4_layouts() {
        for group_size in [16, 32] {
            let metadata = MicrofloatMetadata::new(
                MicrofloatFormat::Mxfp4,
                4,
                group_size,
                MicrofloatLayout::OutputInput,
                2,
                3,
                32,
            )
            .expect("supported MXFP4 metadata");
            assert_eq!(metadata.required_code_bytes(), 96);
            assert_eq!(metadata.required_scale_bytes(), 2 * 3 * 32 / group_size as usize);
        }
        assert!(matches!(
            MicrofloatMetadata::new(MicrofloatFormat::Nvfp4, 4, 16, MicrofloatLayout::OutputInput, 1, 1, 16),
            Err(MicrofloatError::UnsupportedFormat(MicrofloatFormat::Nvfp4))
        ));
    }
}
