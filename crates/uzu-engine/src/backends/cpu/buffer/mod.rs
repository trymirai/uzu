use crate::{
    backends::{
        common::{Backend, Buffer},
        cpu::Cpu,
    },
    utils::downcast::downcast_ref,
};

pub mod dense;
pub mod sparse;

pub(super) trait CpuBufferExt: Buffer<Backend = Cpu> {
    fn downcast(&self) -> &<Cpu as Backend>::GlobalBuffer {
        if let Some(buffer) = downcast_ref::<<Cpu as Backend>::GlobalBuffer>(self) {
            buffer
        } else {
            unreachable!("Unsupported Cpu buffer type")
        }
    }
}

impl<T: Buffer<Backend = Cpu> + ?Sized> CpuBufferExt for T {}
