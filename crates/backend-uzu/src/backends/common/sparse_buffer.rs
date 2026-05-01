use std::ops::Range;

use backend_uzu::backends::common::Backend;

pub struct SparseBufferOperation {
    pub map: bool,
    pub pages: Range<usize>,
    pub heap_page_offset: usize,
}

pub trait SparseBuffer {
    type Backend: Backend<SparseBuffer = Self>;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer;

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );

    fn gpu_ptr(&self) -> usize;

    fn length(&self) -> usize;

    fn execute(
        &self,
        context: &<Self::Backend as Backend>::Context,
        operations: &[SparseBufferOperation],
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
