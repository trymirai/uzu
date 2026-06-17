use std::{os::raw::c_void, ptr::NonNull};

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::{Buffer, DenseBuffer, ResourceHandle};

impl Buffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.gpu_address() as usize
    }

    fn size(&self) -> usize {
        self.length()
    }

    fn resource_handle(&self) -> ResourceHandle {
        NonNull::new(Retained::as_ptr(self) as *mut c_void)
    }
}

impl DenseBuffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.contents()
    }
}
