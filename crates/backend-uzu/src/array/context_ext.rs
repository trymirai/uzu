use crate::{
    Array, ArrayElement, DataType,
    array::size_for_shape,
    backends::common::{AllocationType, Context},
};

pub trait ArrayContextExt: Context {
    fn create_array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> Array<Self::Backend> {
        let size = size_for_shape(shape, data_type);
        let allocation = self.create_allocation(size, AllocationType::Global).expect("Failed to create allocation");
        unsafe { Array::from_allocation(allocation, 0, shape, data_type) }
    }

    fn create_array_zeros(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> Array<Self::Backend> {
        let mut array = self.create_array_uninitialized(shape, data_type);
        array.as_bytes_mut().fill(0);
        array
    }

    fn create_array_from<T: ArrayElement>(
        &self,
        shape: &[usize],
        data: &[T],
    ) -> Array<Self::Backend> {
        let size_from_shape: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size_from_shape,
            "Shape size {} and data size {} are not equal",
            size_from_shape,
            data.len()
        );

        let mut array = self.create_array_uninitialized(shape, T::data_type());
        array.as_slice_mut().copy_from_slice(data);
        array
    }
}

impl<C: Context> ArrayContextExt for C {}
