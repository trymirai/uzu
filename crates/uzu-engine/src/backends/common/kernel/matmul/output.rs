#[cfg(any(backend = "cpu", backend = "metal"))] // TODO: webgpu
use thiserror::Error;

use crate::backends::common::{Backend, BufferMut, kernel::matmul::MatmulDOps};

#[cfg(any(backend = "cpu", backend = "metal"))] // TODO: webgpu
#[derive(Debug, Error)]
#[error("Unsupported output row stride {stride} for {columns} columns on path {path}")]
pub struct MatmulOutputError {
    path: &'static str,
    stride: u32,
    columns: u32,
}

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

    #[cfg(any(backend = "cpu", backend = "metal"))] // TODO: webgpu
    pub(crate) fn into_contiguous(
        self,
        columns: u32,
        path: &'static str,
    ) -> Result<(T, MatmulDOps<'d, B>), MatmulOutputError> {
        let stride = self.row_stride.unwrap_or(columns);
        if stride != columns {
            return Err(MatmulOutputError {
                path,
                stride,
                columns,
            });
        }
        Ok((self.values, self.ops))
    }
}
