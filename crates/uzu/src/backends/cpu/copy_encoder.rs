use std::{ops::Range, ptr};

use crate::backends::{
    common::{Backend, CopyEncoder, NativeBuffer},
    cpu::backend::Cpu,
};

#[derive(Clone)]
pub struct CpuCopyEncoder;

impl CopyEncoder for CpuCopyEncoder {
    type Backend = Cpu;

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    ) {
        assert!(src.length() >= size && dst.length() >= size);
        unsafe { ptr::copy_nonoverlapping(src.data().as_ptr(), dst.cpu_ptr().as_ptr() as *mut u8, size) }
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    ) {
        assert!(range.start <= range.end && range.end <= dst.length());
        unsafe {
            ptr::write_bytes((dst.cpu_ptr().as_ptr() as *mut u8).add(range.start), value, range.end - range.start);
        }
    }
}
