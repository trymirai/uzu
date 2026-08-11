use std::borrow::Cow;

use safetensors::{Dtype, View};

use super::Error;
use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend},
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
