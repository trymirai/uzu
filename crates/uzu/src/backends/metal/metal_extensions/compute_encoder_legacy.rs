use std::ptr::NonNull;

use crate::backends::metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
    ProtocolObject,
};

/// Extension trait providing legacy-style API for compute command encoders.
/// This matches the old metal-rs API signature to minimize migration changes.
pub trait ComputeEncoderLegacy {
    /// Set bytes with legacy parameter order: (index, length, bytes)
    /// New API is: (bytes, length, index)
    fn set_bytes(&self, index: u64, length: u64, bytes: *const std::ffi::c_void);

    /// Set buffer with legacy parameter order: (index, buffer, offset)
    /// New API is: (buffer, offset, index)
    fn set_buffer(
        &self,
        index: u64,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
    );

    /// Dispatch threadgroups (same signature, kept for consistency)
    fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    );

    /// Dispatch threads (same signature, kept for consistency)
    fn dispatch_threads(
        &self,
        threads_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    );

    /// Set compute pipeline state
    fn set_compute_pipeline_state(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
    );
}

impl ComputeEncoderLegacy for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn set_bytes(&self, index: u64, length: u64, bytes: *const std::ffi::c_void) {
        if let Some(ptr) = NonNull::new(bytes as *mut std::ffi::c_void) {
            unsafe {
                MTLComputeCommandEncoder::set_bytes(
                    self,
                    ptr,
                    length as usize,
                    index as usize,
                );
            }
        }
    }

    fn set_buffer(
        &self,
        index: u64,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
    ) {
        MTLComputeCommandEncoder::set_buffer(
            self,
            buffer,
            offset as usize,
            index as usize,
        );
    }

    fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        MTLComputeCommandEncoder::dispatch_threadgroups(
            self,
            threadgroups_per_grid,
            threads_per_threadgroup,
        );
    }

    fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        MTLComputeCommandEncoder::dispatch_threads(
            self,
            threads_per_grid,
            threads_per_threadgroup,
        );
    }

    fn set_compute_pipeline_state(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
    ) {
        MTLComputeCommandEncoder::set_compute_pipeline_state(self, state);
    }
}

/// Blanket implementation for references to avoid double-reference issues
impl<T: ComputeEncoderLegacy + ?Sized> ComputeEncoderLegacy for &T {
    fn set_bytes(&self, index: u64, length: u64, bytes: *const std::ffi::c_void) {
        (*self).set_bytes(index, length, bytes)
    }

    fn set_buffer(
        &self,
        index: u64,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
    ) {
        (*self).set_buffer(index, buffer, offset)
    }

    fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        (*self).dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup)
    }

    fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        (*self).dispatch_threads(threads_per_grid, threads_per_threadgroup)
    }

    fn set_compute_pipeline_state(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
    ) {
        (*self).set_compute_pipeline_state(state)
    }
}
