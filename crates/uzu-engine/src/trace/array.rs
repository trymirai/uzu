use std::borrow::Cow;

use bytemuck::{AnyBitPattern, NoUninit};
use safetensors::{Dtype, View};

use crate::{
    array::size_for_shape,
    backends::common::{Allocation, AllocationType, Backend, Context, Encoder},
    data_type::DataType,
};

pub struct Array<B: Backend> {
    shape: Box<[u32]>,
    // safetensors' View wants usize, so keep a converted copy rather than rebuild it per call
    view_shape: Box<[usize]>,
    data_type: DataType,
    dtype: Dtype,
    allocation: Allocation<B>,
}

impl<B: Backend> Array<B> {
    pub fn new(
        shape: &[u32],
        data_type: DataType,
        allocation: Allocation<B>,
    ) -> Self {
        Self {
            shape: shape.into(),
            view_shape: shape.iter().map(|dim| *dim as usize).collect(),
            data_type,
            dtype: data_type.try_into().expect("data type has a safetensors equivalent"),
            allocation,
        }
    }

    pub fn capture(
        allocation: &Allocation<B>,
        shape: &[u32],
        data_type: DataType,
        encoder: &mut Encoder<B>,
    ) -> Result<Self, B::Error> {
        let byte_count = size_for_shape(shape, data_type);
        assert!(allocation.size() >= byte_count, "capture declares more bytes than the source allocation holds");

        let mut destination = encoder.context.create_allocation(byte_count, AllocationType::Global)?;
        encoder.encode_copy(allocation, ..byte_count, &mut destination, ..);

        Ok(Self::new(shape, data_type, destination))
    }

    pub fn capture_slice<T: NoUninit + AnyBitPattern>(
        data: &[T],
        shape: &[u32],
        data_type: DataType,
        encoder: &Encoder<B>,
    ) -> Result<Self, B::Error> {
        let byte_count = size_for_shape(shape, data_type);
        assert_eq!(byte_count, std::mem::size_of_val(data), "capture_slice shape does not match the data");

        let mut destination = encoder.context.create_allocation(byte_count, AllocationType::Global)?;
        destination.copyin(data);

        Ok(Self::new(shape, data_type, destination))
    }
}

impl<B: Backend> View for &Array<B> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.view_shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.allocation.as_slice::<u8>()[..self.data_len()])
    }

    fn data_len(&self) -> usize {
        size_for_shape(&self.shape, self.data_type)
    }
}
