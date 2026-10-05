use crate::{
    backends::{
        amdgpu::Amdgpu,
        common::{Backend, Buffer},
    },
    utils::downcast::downcast_ref,
};

pub mod constant;
pub mod dense;
pub mod device;
pub mod scratch;
pub mod sparse;

/// Device address of any AMDGPU buffer (global, constant or scratch allocation).
pub trait AmdgpuBufferExt: Buffer<Backend = Amdgpu> {
    fn device_address(&self) -> u64 {
        if let Some(buffer) = downcast_ref::<<Amdgpu as Backend>::GlobalBuffer>(self) {
            buffer.device_address()
        } else if let Some(buffer) = downcast_ref::<<Amdgpu as Backend>::ConstantBuffer>(self) {
            buffer.page().device_address() + buffer.range().start as u64
        } else if let Some(buffer) = downcast_ref::<<Amdgpu as Backend>::ScratchBuffer>(self) {
            buffer.page().device_address() + buffer.range().start as u64
        } else {
            unreachable!("Unsupported AMDGPU buffer type")
        }
    }
}

impl<T: Buffer<Backend = Amdgpu> + ?Sized> AmdgpuBufferExt for T {}
