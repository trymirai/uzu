use metal::{
    CommandBuffer, CommandBufferRef, ComputeCommandEncoder,
    ComputeCommandEncoderRef, Device, foreign_types::ForeignType,
};
use objc2::{msg_send, runtime::AnyObject};

pub trait CommandBufferTimingAccess {
    fn kernel_start_time(&self) -> Option<f64>;
    fn kernel_end_time(&self) -> Option<f64>;
    fn gpu_execution_time_ms(&self) -> Option<f64>;
}

impl CommandBufferTimingAccess for CommandBuffer {
    fn kernel_start_time(&self) -> Option<f64> {
        unsafe {
            let ptr = self.as_ptr();
            let obj = ptr as *mut AnyObject;
            let start_time: f64 = msg_send![obj, kernelStartTime];
            if start_time > 0.0 {
                Some(start_time)
            } else {
                None
            }
        }
    }

    fn kernel_end_time(&self) -> Option<f64> {
        unsafe {
            let ptr = self.as_ptr();
            let obj = ptr as *mut AnyObject;
            let end_time: f64 = msg_send![obj, kernelEndTime];
            if end_time > 0.0 {
                Some(end_time)
            } else {
                None
            }
        }
    }

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        if let (Some(start), Some(end)) =
            (self.kernel_start_time(), self.kernel_end_time())
        {
            Some((end - start) * 1000.0)
        } else {
            None
        }
    }
}

impl CommandBufferTimingAccess for CommandBufferRef {
    fn kernel_start_time(&self) -> Option<f64> {
        unsafe {
            let obj = self as *const _ as *mut AnyObject;
            let start_time: f64 = msg_send![obj, kernelStartTime];
            if start_time > 0.0 {
                Some(start_time)
            } else {
                None
            }
        }
    }

    fn kernel_end_time(&self) -> Option<f64> {
        unsafe {
            let obj = self as *const _ as *mut AnyObject;
            let end_time: f64 = msg_send![obj, kernelEndTime];
            if end_time > 0.0 {
                Some(end_time)
            } else {
                None
            }
        }
    }

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        if let (Some(start), Some(end)) =
            (self.kernel_start_time(), self.kernel_end_time())
        {
            Some((end - start) * 1000.0)
        } else {
            None
        }
    }
}
