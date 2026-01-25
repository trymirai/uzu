use crate::backends::vulkan::buffer::VkBuffer;
use crate::{array_size_in_bytes, Array, DataType};

pub struct VkArray {
    buffer: VkBuffer,
    shape: Box<[usize]>,
    data_type: DataType,
    size: usize,
    offset: usize,
    label: String,
}

impl VkArray {
    pub fn new(
        buffer: VkBuffer,
        shape: &[usize],
        data_type: DataType
    ) -> Self {
        Self::new_with_offset_and_label(buffer, shape, data_type, 0, "".to_string())
    }

    pub fn new_with_offset_and_label(
        buffer: VkBuffer,
        shape: &[usize],
        data_type: DataType,
        offset: usize,
        label: String
    ) -> Self {
        let size = array_size_in_bytes(&shape, data_type);
        Self { buffer, shape: shape.into(), data_type, size, offset, label }
    }
}

impl Array for VkArray {
    type BackendBuffer = VkBuffer;

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }

    fn label(&self) -> String {
        String::from(&self.label)
    }

    fn buffer(&self) -> &[u8] {
        self.buffer.get_bytes().unwrap()
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        self.buffer.get_bytes_mut().unwrap()   
    }

    fn backend_buffer(&self) -> &Self::BackendBuffer {
        &self.buffer
    }
}