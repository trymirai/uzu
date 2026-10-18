use crate::backends::{
    common::{
        BufferMut, BufferRef, CommandBufferEncoding, Kernels,
        kernel::{Unsupported, radix_top_k_small::RadixTopKSmall},
    },
    webgpu::{
        WebGPU, buffer::binding_parts, command_buffer::WebGPUCommandBufferEncoding, context::WebGPUContext,
        error::WebGPUError,
    },
};

mod attention;
mod matmul;

include!(concat!(env!("OUT_DIR"), "/webgpu.rs"));

impl Kernels for WebGPUKernels {
    type Backend = WebGPU;

    autogen_kernels!();
    type AttentionKernel = attention::WebGPUAttention;
    type DeltaNetChunkedPrefill = Unsupported<WebGPU>;
    type DeltaNetTreeVerify = Unsupported<WebGPU>;
    type MatmulKernel = matmul::WebGPUMatmul;
    type RadixTopKSmall = Unsupported<WebGPU>;
}

impl RadixTopKSmall for Unsupported<WebGPU> {
    type Backend = WebGPU;

    fn new(
        _context: &WebGPUContext,
        _columns: u32,
    ) -> Result<Self, WebGPUError> {
        Err(WebGPUError::NotSupported)
    }

    fn encode(
        &self,
        _input: impl BufferRef<Backend = WebGPU>,
        _output_ids: impl BufferMut<Backend = WebGPU>,
        _output_scores: impl BufferMut<Backend = WebGPU>,
        _rows: u32,
        _k: u32,
        _command_buffer: &mut WebGPUCommandBufferEncoding,
    ) -> Result<(), WebGPUError> {
        Err(WebGPUError::NotSupported)
    }
}
