use metal::MTLSparsePageSize;

pub trait SparsePageSizeExt {
    fn in_bytes(&self) -> usize;
}

impl SparsePageSizeExt for MTLSparsePageSize {
    fn in_bytes(&self) -> usize {
        match self {
            MTLSparsePageSize::KB16 => 16 * 1024,
            MTLSparsePageSize::KB64 => 64 * 1024,
            MTLSparsePageSize::KB256 => 256 * 1024,
        }
    }
}
