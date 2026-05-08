use std::{cell::RefCell, rc::Rc};

use crate::{
    Array, ArrayElement, DataType,
    array::size_for_shape,
    backends::common::{Backend, Buffer, Context},
};

pub trait ArrayContextExt: Context {
    fn create_array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> Array<Self::Backend, <Self::Backend as Backend>::DenseBuffer> {
        let buffer_size_bytes = size_for_shape(shape, data_type);

        let mut buffer = self.create_buffer(buffer_size_bytes).expect("Failed to create buffer");
        buffer.set_label(Some(label));

        unsafe { Array::from_parts(Rc::new(RefCell::new(buffer)), 0, shape, data_type) }
    }

    fn create_array_zeros(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: &str,
    ) -> Array<Self::Backend, <Self::Backend as Backend>::DenseBuffer> {
        let mut array = self.create_array_uninitialized(shape, data_type, label);
        array.as_bytes_mut().fill(0);
        array
    }

    fn create_array_from<T: ArrayElement>(
        &self,
        shape: &[usize],
        data: &[T],
        label: &str,
    ) -> Array<Self::Backend, <Self::Backend as Backend>::DenseBuffer> {
        let size_from_shape: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size_from_shape,
            "Shape size {} and data size {} are not equal",
            size_from_shape,
            data.len()
        );

        let mut array = self.create_array_uninitialized(shape, T::data_type(), label);
        array.as_slice_mut().copy_from_slice(data);
        array
    }
}

impl<C: Context> ArrayContextExt for C {}
