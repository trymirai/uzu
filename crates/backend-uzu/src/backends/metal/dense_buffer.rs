use std::{ops::Range, os::raw::c_void, ptr::NonNull};

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::{
    common::{Backend, Buffer, DenseBuffer},
    metal::error::MetalError,
};

impl Buffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;

    fn as_bytes_slice_range(
        &self,
        _context: Option<&<Self::Backend as Backend>::Context>,
        range: Range<usize>,
    ) -> Result<&[u8], MetalError> {
        assert!(range.end <= self.length());
        let start_ptr = unsafe { self.cpu_ptr().add(range.start) };
        let slice = unsafe { std::slice::from_raw_parts(start_ptr.as_ptr() as *const u8, range.len()) };
        Ok(slice)
    }

    fn gpu_ptr(&self) -> usize {
        self.gpu_address() as usize
    }

    fn size(&self) -> usize {
        (**self).length()
    }
}

impl DenseBuffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.contents()
    }
}
