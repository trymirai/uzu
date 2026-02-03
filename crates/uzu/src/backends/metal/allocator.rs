use std::ops::Deref;

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::NativeBuffer;

use super::Metal;

impl NativeBuffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;

    fn length(&self) -> usize {
        self.deref().length()
    }

    fn id(&self) -> usize {
        Retained::as_ptr(self) as usize
    }
}
