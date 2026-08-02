use std::range::Range;

use crate::{
    backends::{
        common::{Backend, GpuBuffer},
        webgpu::WebGPU,
    },
    utils::downcast::downcast_ref,
};

pub mod dense;
pub mod sparse;

pub fn binding_parts(
    buffer: &(impl GpuBuffer<Backend = WebGPU> + ?Sized),
    range: Range<usize>,
) -> (&wgpu::Buffer, usize, usize) {
    let (buffer, offset) = if let Some(buffer) = downcast_ref::<<WebGPU as Backend>::GlobalGpuBuffer>(buffer) {
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
