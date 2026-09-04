use super::{MicrofloatEncoding, MicrofloatError};

/// Physical shape and derived strides for one microfloat matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatMetadata {
    pub encoding: MicrofloatEncoding,
    pub rows: u32,
    pub columns: u32,
}

impl MicrofloatMetadata {
    pub fn new(
        encoding: MicrofloatEncoding,
        rows: u32,
        columns: u32,
    ) -> Result<Self, MicrofloatError> {
        let metadata = Self {
            encoding,
            rows,
            columns,
        };
        metadata.storage_sizes()?;
        Ok(metadata)
    }

    pub fn code_row_stride(self) -> usize {
        self.columns as usize / 2
    }

    pub fn scale_row_stride(self) -> usize {
        self.columns as usize / self.encoding.group_size as usize
    }

    /// Validate the encoding and shape, then return code and scale byte counts.
    pub fn storage_sizes(self) -> Result<(usize, usize), MicrofloatError> {
        self.encoding.validate()?;
        if self.rows == 0 || self.columns == 0 {
            return Err(MicrofloatError::EmptyShape);
        }
        if !self.columns.is_multiple_of(self.encoding.group_size) {
            return Err(MicrofloatError::MisalignedColumns {
                columns: self.columns,
                group_size: self.encoding.group_size,
            });
        }
        let codes = (self.rows as usize).checked_mul(self.code_row_stride()).ok_or(MicrofloatError::SizeOverflow)?;
        let scales = (self.rows as usize).checked_mul(self.scale_row_stride()).ok_or(MicrofloatError::SizeOverflow)?;
        Ok((codes, scales))
    }
}
