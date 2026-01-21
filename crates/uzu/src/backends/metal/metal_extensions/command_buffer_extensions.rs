use crate::backends::metal::{MTLCommandBuffer, ProtocolObject};
use objc2::msg_send;

pub trait CommandBufferTimingAccess {
    fn kernel_start_time(&self) -> Option<f64>;
    fn kernel_end_time(&self) -> Option<f64>;
    fn gpu_execution_time_ms(&self) -> Option<f64>;
}

impl CommandBufferTimingAccess for ProtocolObject<dyn MTLCommandBuffer> {
    fn kernel_start_time(&self) -> Option<f64> {
        unsafe {
            let start_time: f64 = msg_send![self, kernelStartTime];
            if start_time > 0.0 {
                Some(start_time)
            } else {
                None
            }
        }
    }

    fn kernel_end_time(&self) -> Option<f64> {
        unsafe {
            let end_time: f64 = msg_send![self, kernelEndTime];
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
