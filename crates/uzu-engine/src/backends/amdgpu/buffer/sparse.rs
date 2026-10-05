use std::range::Range;

use crate::backends::{
    amdgpu::{Amdgpu, error::AmdgpuError},
    common::{Backend, Buffer, SparseBuffer},
};

/// Sparse buffers are not supported yet (`DeviceCapabilities::empty()`); the engine falls back to
/// dense KV caches. `AmdgpuContext::create_sparse_buffer` never returns one.
#[derive(Debug)]
pub struct AmdgpuSparseBuffer {
    never: std::convert::Infallible,
}

impl Buffer for AmdgpuSparseBuffer {
    type Backend = Amdgpu;

    fn size(&self) -> usize {
        match self.never {}
    }
}

impl SparseBuffer for AmdgpuSparseBuffer {
    fn map(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _pages: impl Into<Range<usize>>,
    ) -> Result<(), AmdgpuError> {
        match self.never {}
    }

    fn unmap(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _pages: impl Into<Range<usize>>,
    ) -> Result<(), AmdgpuError> {
        match self.never {}
    }

    fn page_size_bytes(&self) -> usize {
        match self.never {}
    }
}
