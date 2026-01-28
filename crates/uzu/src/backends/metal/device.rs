use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::{AllocError, Device};

impl Device for Retained<ProtocolObject<dyn MTLDevice>> {
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type ResourceOptions = MTLResourceOptions;

    fn create_buffer(
        &self,
        size: usize,
        options: MTLResourceOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, AllocError> {
        self.new_buffer(size, options).ok_or_else(|| {
            AllocError::AllocationFailed {
                size,
                reason: "device.newBuffer returned nil".to_string(),
            }
        })
    }
}
