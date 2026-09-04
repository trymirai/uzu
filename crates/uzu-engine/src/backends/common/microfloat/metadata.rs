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
        if rows == 0 || columns == 0 {
            return Err(MicrofloatError::EmptyShape);
        }
        if !columns.is_multiple_of(encoding.group_size) {
            return Err(MicrofloatError::MisalignedColumns {
                columns,
                group_size: encoding.group_size,
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

    pub fn code_row_stride(self) -> usize {
        self.columns as usize / 2
    }

    pub fn scale_row_stride(self) -> usize {
        self.columns as usize / self.encoding.group_size as usize
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
