use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, Kernels,
            kernel::{
                AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, Unsupported,
                matmul::{MatmulArguments, MatmulKernel},
                radix_top_k_small::RadixTopKSmall,
            },
        },
        webgpu::{
            WebGPU, buffer::binding_parts, command_buffer::WebGPUCommandBufferEncoding, context::WebGPUContext,
            error::WebGPUError,
        },
    },
    data_type::DataType,
};

include!(concat!(env!("OUT_DIR"), "/webgpu.rs"));

impl Kernels for WebGPUKernels {
    type Backend = WebGPU;

    autogen_kernels!();
    type AttentionKernel = Unsupported<WebGPU>;
    type DeltaNetChunkedPrefill = Unsupported<WebGPU>;
    type DeltaNetTreeVerify = Unsupported<WebGPU>;
    type MatmulKernel = Unsupported<WebGPU>;
    type RadixTopKSmall = Unsupported<WebGPU>;
}

impl AttentionKernel for Unsupported<WebGPU> {
    type Backend = WebGPU;
    type Error = WebGPUError;

    fn new(
        _context: &WebGPUContext,
        _config: AttentionKernelConfig,
    ) -> Result<Self, WebGPUError> {
        Err(WebGPUError::NotSupported)
    }

    fn encode(
        &self,
        _input: AttentionInput<
            impl BufferRef<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
        >,
        _arguments: AttentionArguments<'_, WebGPU, impl BufferRef<Backend = WebGPU>, impl BufferRef<Backend = WebGPU>>,
        _command_buffer: &mut WebGPUCommandBufferEncoding,
    ) -> Result<<WebGPU as Backend>::ScratchBuffer, WebGPUError> {
        Err(WebGPUError::NotSupported)
    }
}

impl MatmulKernel for Unsupported<WebGPU> {
    type Backend = WebGPU;
    type Error = WebGPUError;
    type WeightsLayout = ();

    fn new(
        _context: &WebGPUContext,
        _weights_data_type: DataType,
        _input_data_type: DataType,
        _output_data_type: DataType,
    ) -> Result<Self, WebGPUError> {
        Err(WebGPUError::NotSupported)
    }

    fn encode(
        &mut self,
        _arguments: MatmulArguments<
            '_,
            WebGPU,
            impl BufferRef<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
            impl BufferMut<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
        >,
        _command_buffer: &mut WebGPUCommandBufferEncoding,
    ) -> Result<(), WebGPUError> {
        Err(WebGPUError::NotSupported)
    }
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

#[cfg(test)]
#[path = "../../../unit/backends/webgpu/kernel/mod.rs"]
mod tests;
