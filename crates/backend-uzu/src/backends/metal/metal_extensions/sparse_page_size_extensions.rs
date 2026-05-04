use bytesize::ByteSize;
use metal::MTLSparsePageSize;

pub trait SparsePageSizeExt {
    fn byte_size(&self) -> ByteSize;
}

impl SparsePageSizeExt for MTLSparsePageSize {
    fn byte_size(&self) -> ByteSize {
        match self {
            MTLSparsePageSize::KB16 => ByteSize::kib(16),
            MTLSparsePageSize::KB64 => ByteSize::kib(64),
            MTLSparsePageSize::KB256 => ByteSize::kib(256),
        }
    }
}
