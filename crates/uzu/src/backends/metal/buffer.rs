use std::{os::raw::c_void, ptr::NonNull};

use metal::{BufferExt, MTLBuffer, MTLResourceExt};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::Buffer;

impl Buffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        (**self).set_label(label);
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.contents()
    }

    fn gpu_ptr(&self) -> usize {
        self.gpu_address() as usize
    }

    fn length(&self) -> usize {
        (**self).length()
    }

    fn did_modify_range(
        &self,
        range: std::ops::Range<usize>,
    ) {
        (**self).did_modify_range(range);
    }
}
