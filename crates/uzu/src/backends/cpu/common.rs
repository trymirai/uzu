use super::CPUBackend;
use crate::{Array, DataType, array::array_size_in_bytes, backends::Context};

#[derive(Debug, Clone)]
pub struct CPUArray {
    buffer: Box<[u8]>,
    shape: Box<[usize]>,
    data_type: DataType,
}

impl Array for CPUArray {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }

    fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    fn copy_slice(
        &mut self,
        _source: &Self,
        _axis: usize,
        _src_range: std::ops::Range<usize>,
        _dst_offset: usize,
    ) {
        panic!("not implemented") // TODO: implement it :)
    }
}

impl CPUArray {
    /// Create a new Array
    pub fn new(
        shape: &[usize],
        data_type: DataType,
    ) -> Self {
        let buffer_size_bytes = array_size_in_bytes(shape, data_type);
        let buffer =
            std::iter::repeat(0 as u8).take(buffer_size_bytes).collect();
        Self {
            buffer,
            shape: shape.into(),
            data_type,
        }
    }
}

pub struct CPUContext {}

impl CPUContext {
    pub fn new() -> Self {
        Self {}
    }
}

impl Context<CPUBackend> for CPUContext {
    fn default() -> Option<Self> {
        Some(Self {})
    }
    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> CPUArray {
        CPUArray::new(shape, data_type)
    }
}
