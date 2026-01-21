use crate::backends::metal::{Device, MTLComputePipelineState, ProtocolObject};

/// Extensions for ComputePipelineState to get the device
pub trait ComputePipelineStateDeviceAccess {
    /// Gets the device associated with this compute pipeline state.
    fn device(&self) -> Device;
}

impl ComputePipelineStateDeviceAccess for ProtocolObject<dyn MTLComputePipelineState> {
    fn device(&self) -> Device {
        // Use the MTLComputePipelineState::device() method from mtl-rs
        MTLComputePipelineState::device(self)
    }
}
