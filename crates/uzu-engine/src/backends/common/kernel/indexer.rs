#[cfg(any(backend = "cpu", backend = "vulkan", backend = "webgpu"))]
use crate::backends::common::kernel::Unsupported;
use crate::backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, Kernels, kernel::attention::KeyBlocks};

pub trait IndexerKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<IndexerKernel = Self>>;

    fn is_supported(context: &<Self::Backend as Backend>::Context) -> bool;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        num_heads: u32,
        head_dim: u32,
        compression_ratio: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn prepare_draft_blocks(
        &self,
        trie: Option<impl BufferRef<Backend = Self::Backend>>,
        rows: u32,
        prefix: u32,
        draft_token_rows: impl BufferMut<Backend = Self::Backend>,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, <Self::Backend as Backend>::Error>;

    fn select_blocks(
        &self,
        query_codes: impl BufferRef<Backend = Self::Backend>,
        query_scales: impl BufferRef<Backend = Self::Backend>,
        key_codes: impl BufferRef<Backend = Self::Backend>,
        key_scales: impl BufferRef<Backend = Self::Backend>,
        rows: u32,
        prefix_len: u32,
        cached_block_row: u32,
        candidates: u32,
        selected: KeyBlocks<impl BufferMut<Backend = Self::Backend>>,
        trie: Option<impl BufferRef<Backend = Self::Backend>>,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

#[cfg(any(backend = "cpu", backend = "vulkan", backend = "webgpu"))]
impl<B: Backend<Kernels: Kernels<IndexerKernel = Unsupported<B>>>> IndexerKernel for Unsupported<B> {
    type Backend = B;

    fn is_supported(_context: &B::Context) -> bool {
        false
    }

    fn new(
        _context: &B::Context,
        _num_heads: u32,
        _head_dim: u32,
        _compression_ratio: u32,
    ) -> Result<Self, B::Error> {
        unreachable!("unsupported indexer should not be constructed")
    }

    fn prepare_draft_blocks(
        &self,
        _trie: Option<impl BufferRef<Backend = B>>,
        _rows: u32,
        _prefix: u32,
        _draft_token_rows: impl BufferMut<Backend = B>,
        _command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        unreachable!("unsupported indexer cannot encode")
    }

    fn select_blocks(
        &self,
        _query_codes: impl BufferRef<Backend = B>,
        _query_scales: impl BufferRef<Backend = B>,
        _key_codes: impl BufferRef<Backend = B>,
        _key_scales: impl BufferRef<Backend = B>,
        _rows: u32,
        _prefix_len: u32,
        _cached_block_row: u32,
        _candidates: u32,
        _selected: KeyBlocks<impl BufferMut<Backend = B>>,
        _trie: Option<impl BufferRef<Backend = B>>,
        _command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        unreachable!("unsupported indexer cannot encode")
    }
}
