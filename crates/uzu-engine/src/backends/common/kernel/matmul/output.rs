use super::{MatmulDOps, MatmulError};
use crate::backends::common::{Backend, BufferMut};

pub struct MatmulOutput<'d, B: Backend, T: BufferMut<Backend = B>> {
    pub values: T,
    /// Elements between output rows; `None` means the logical output width `n`.
    pub row_stride: Option<u32>,
    pub ops: MatmulDOps<'d, B>,
}

impl<'d, B: Backend, T: BufferMut<Backend = B>> MatmulOutput<'d, B, T> {
    pub fn new(
        values: T,
        ops: MatmulDOps<'d, B>,
    ) -> Self {
        Self {
            values,
            row_stride: None,
            ops,
        }
    }

    pub(crate) fn into_contiguous(
        self,
        columns: u32,
        path: &'static str,
    ) -> Result<(T, MatmulDOps<'d, B>), MatmulError<B>> {
        let stride = self.row_stride.unwrap_or(columns);
        if stride != columns {
            return Err(MatmulError::UnsupportedOutputStride {
                path,
                stride,
                columns,
            });
        }
        Ok((self.values, self.ops))
    }
}
