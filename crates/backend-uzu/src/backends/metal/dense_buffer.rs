use std::{os::raw::c_void, ptr::NonNull};

use metal::{MTLBuffer, MTLResourceExt};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::{Buffer, DenseBuffer};

impl Buffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.gpu_address() as usize
    }

    fn size(&self) -> usize {
        (**self).length()
    }

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        (**self).set_label(label);
    }
}

impl DenseBuffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.contents()
    }
}
