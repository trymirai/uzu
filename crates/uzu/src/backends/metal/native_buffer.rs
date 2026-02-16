use std::{ops::Deref, os::raw::c_void, ptr::NonNull};

use metal::{MTLBuffer, MTLResourceExt};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::NativeBuffer;

impl NativeBuffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;

    fn set_label(
        &self,
        label: Option<&str>,
    ) {
        MTLResourceExt::set_label(self.deref(), label);
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.contents()
    }

    fn length(&self) -> usize {
        self.deref().length()
    }

    fn id(&self) -> usize {
        Retained::as_ptr(self) as usize
    }
}
