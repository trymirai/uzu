#[cfg(any(backend = "cpu", backend = "metal"))] // TODO: webgpu
use thiserror::Error;

#[cfg(any(backend = "cpu", backend = "metal"))] // TODO: webgpu
use crate::backends::common::gpu_types::gemm::GemmDTransform;
use crate::backends::common::{Backend, BufferMut, kernel::matmul::MatmulDOps};

#[cfg(any(backend = "cpu", backend = "metal"))] // TODO: webgpu
#[derive(Debug, Error)]
#[error("Unsupported output row stride {stride} for {columns} columns with D ops {ops:?}")]
pub struct MatmulOutputError {
    stride: u32,
    columns: u32,
    ops: GemmDTransform,
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
    pub fn checked_row_stride(
        &self,
        columns: u32,
    ) -> Result<u32, MatmulOutputError> {
        let stride = self.row_stride.unwrap_or(columns);
        let ops = self.ops.mask();
        if stride < columns || (stride != columns && !ops.is_empty()) {
            return Err(MatmulOutputError {
                stride,
                columns,
                ops,
            });
        }
        Ok(stride)
    }
}
