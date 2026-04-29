use crate::backends::{common::Event, webgpu::WebGPU};

pub struct WebGPUEvent {}

impl Event for WebGPUEvent {
    type Backend = WebGPU;
}
