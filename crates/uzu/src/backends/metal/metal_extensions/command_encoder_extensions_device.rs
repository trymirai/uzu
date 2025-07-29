use metal::{
    ComputeCommandEncoder, ComputeCommandEncoderRef, Device,
    foreign_types::ForeignType,
};
use objc2::{msg_send, runtime::AnyObject};
pub trait CommandEncoderDeviceAccess {
    fn device(&self) -> Device;
}

impl CommandEncoderDeviceAccess for ComputeCommandEncoder {
    fn device(&self) -> Device {
        unsafe {
            let ptr = self.as_ptr();
            let obj = ptr as *mut AnyObject;
            let device_ptr: *mut AnyObject = msg_send![obj, device];
            Device::from_ptr(device_ptr as *mut metal::MTLDevice)
        }
    }
}

impl CommandEncoderDeviceAccess for ComputeCommandEncoderRef {
    fn device(&self) -> Device {
        unsafe {
            let obj = self as *const _ as *mut AnyObject;
            let device_ptr: *mut AnyObject = msg_send![obj, device];
            Device::from_ptr(device_ptr as *mut metal::MTLDevice)
        }
    }
}
