use crate::{Array, DataType, DeviceContext, array::array_size_in_bytes};

#[derive(Debug, Clone)]
pub struct CPUArray {
    buffer: Box<[u8]>,
    shape: Box<[usize]>,
    data_type: DataType,
    label: String,
}

impl Array for CPUArray {
    type BackendBuffer = Box<[u8]>;

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn label(&self) -> String {
        self.label.clone()
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

    fn backend_buffer(&self) -> &Box<[u8]> {
        &self.buffer
    }
}

impl CPUArray {
    /// Create a new Array
    pub fn new(
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> Self {
        let buffer_size_bytes = array_size_in_bytes(shape, data_type);
        let buffer =
            std::iter::repeat(0 as u8).take(buffer_size_bytes).collect();
        Self {
            buffer,
            shape: shape.into(),
            data_type,
            label,
        }
    }
}

pub struct CPUContext {}

impl CPUContext {
    pub fn new() -> Self {
        Self {}
    }
}

impl DeviceContext for CPUContext {
    type DeviceArray = CPUArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> CPUArray {
        CPUArray::new(shape, data_type, label)
    }
}
