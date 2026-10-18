use std::range::Range;

use crate::{
    backends::{
        common::{Backend, Buffer},
        webgpu::WebGPU,
    },
    utils::downcast::downcast_ref,
};

pub mod constant;
pub mod global;
pub mod scratch;
pub mod sparse;

pub fn binding_parts<'a>(
    buffer: &'a (impl Buffer<Backend = WebGPU> + ?Sized),
    range: Range<usize>,
) -> (&'a wgpu::Buffer, usize, usize) {
    let (buffer, offset) = if let Some(buffer) = downcast_ref::<<WebGPU as Backend>::GlobalBuffer>(buffer) {
        (buffer.wgpu_buffer(), 0)
    } else if let Some(buffer) = downcast_ref::<<WebGPU as Backend>::ConstantBuffer>(buffer) {
        (buffer.page().wgpu_buffer(), buffer.range().start)
    } else if let Some(buffer) = downcast_ref::<<WebGPU as Backend>::ScratchBuffer>(buffer) {
        (buffer.page().wgpu_buffer(), buffer.range().start)
    } else {
        unreachable!()
    };
    (buffer, offset + range.start, range.iter().len())
}
