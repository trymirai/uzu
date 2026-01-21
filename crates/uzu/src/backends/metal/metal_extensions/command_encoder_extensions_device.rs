use crate::backends::metal::{Device, MTLComputeCommandEncoder, ProtocolObject};
use metal::MTLCommandEncoder;

pub trait CommandEncoderDeviceAccess {
    fn device(&self) -> Device;
}

impl CommandEncoderDeviceAccess for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn device(&self) -> Device {
        // Use the MTLCommandEncoder::device() method from mtl-rs
        unsafe { MTLCommandEncoder::device(self) }
    }
}
