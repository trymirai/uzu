use crate::backends::metal::{MTLComputeCommandEncoder, MTLDevice, ProtocolObject, Retained};
use metal::MTLCommandEncoder;

pub trait CommandEncoderDeviceAccess {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
}

impl CommandEncoderDeviceAccess for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        // Use the MTLCommandEncoder::device() method from mtl-rs
        unsafe { MTLCommandEncoder::device(self) }
    }
}
