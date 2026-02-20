use std::ops::Range;

use crate::backends::{
    common::{Backend, CopyEncoder},
    cpu::backend::Cpu,
};

pub struct CpuCopyEncoder;

impl CopyEncoder for CpuCopyEncoder {
    type Backend = Cpu;

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    ) {
        todo!()
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    ) {
        todo!()
    }
}
