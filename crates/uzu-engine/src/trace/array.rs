use std::borrow::Cow;

use bytemuck::{AnyBitPattern, NoUninit};
use safetensors::{Dtype, View};

use super::Error;
use crate::{
    array::size_for_shape,
    backends::common::{Allocation, AllocationType, Backend, Context, Encoder},
    data_type::DataType,
};

pub struct Array<B: Backend> {
    shape: Box<[usize]>,
    data_type: DataType,
    dtype: Dtype,
    allocation: Allocation<B>,
}

impl<B: Backend> Array<B> {
    pub fn new(
        shape: Box<[usize]>,
        data_type: DataType,
        allocation: Allocation<B>,
    ) -> Result<Self, Error> {
        Ok(Self {
            shape,
            data_type,
            dtype: data_type.try_into()?,
            allocation,
        })
    }

    /// Copies `src` into a fresh global allocation. The copy goes through
    /// [`Encoder::encode_copy`], so the hazard tracker orders it after whatever
    /// kernel produced `src`. A global allocation is required because encode-chain
    /// allocations are moved along and their ranges recycled once dropped.
    pub fn capture(
        encoder: &mut Encoder<B>,
        src: &Allocation<B>,
        shape: &[usize],
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let byte_count = size_for_shape(shape, data_type);
        assert!(src.size() >= byte_count, "capture declares more bytes than the source allocation holds");

        let mut destination = encoder.context().create_allocation(byte_count, AllocationType::Global)?;
        encoder.encode_copy(src, ..byte_count, &mut destination, ..);

        Ok(Self::expect_new(shape, data_type, destination))
    }

    /// For arrays that exist only on the host, such as the i32 token ids uzu feeds
    /// the decoder as u32.
    pub fn capture_host<T: NoUninit + AnyBitPattern>(
        encoder: &Encoder<B>,
        data: &[T],
        shape: &[usize],
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let byte_count = size_for_shape(shape, data_type);
        assert_eq!(byte_count, std::mem::size_of_val(data), "capture_host shape does not match the data");

        let mut destination = encoder.context().create_allocation(byte_count, AllocationType::Global)?;
        destination.copyin(data);

        Ok(Self::expect_new(shape, data_type, destination))
    }

    // Only I4/U4 lack a safetensors dtype, and activations are never either.
    fn expect_new(
        shape: &[usize],
        data_type: DataType,
        allocation: Allocation<B>,
    ) -> Self {
        Self::new(shape.into(), data_type, allocation).expect("activation dtype has a safetensors equivalent")
    }
}

impl<B: Backend> View for &Array<B> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.allocation.as_slice::<u8>()[..self.data_len()])
    }

    fn data_len(&self) -> usize {
        size_for_shape(&self.shape, self.data_type)
    }
}
