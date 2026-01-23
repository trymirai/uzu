use crate::backends::metal::{MTLComputePipelineState, MTLDevice, ProtocolObject, Retained};

/// Extensions for ComputePipelineState to get the device
pub trait ComputePipelineStateDeviceAccess {
    /// Gets the device associated with this compute pipeline state.
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
}

impl ComputePipelineStateDeviceAccess for ProtocolObject<dyn MTLComputePipelineState> {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        // Use the MTLComputePipelineState::device() method from mtl-rs
        MTLComputePipelineState::device(self)
    }
}
