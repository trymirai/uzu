use metal::{ComputePipelineState, Device, foreign_types::ForeignType};
use objc2::{msg_send, runtime::AnyObject};

/// Extensions for ComputePipelineState to get the device
pub trait ComputePipelineStateDeviceAccess {
    /// Gets the device associated with this compute pipeline state.
    fn device(&self) -> Device;
}

impl ComputePipelineStateDeviceAccess for ComputePipelineState {
    fn device(&self) -> Device {
        unsafe {
            let ptr = self.as_ptr();
            let obj = ptr as *mut AnyObject;
            let device_ptr: *mut AnyObject = msg_send![obj, device];
            Device::from_ptr(device_ptr as *mut metal::MTLDevice)
        }
    }
}
