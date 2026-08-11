use std::{collections::HashMap, path::Path};

use safetensors::serialize_to_file;

use super::{Array, Error};
use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

pub struct Recorder<B: Backend> {
    arrays: HashMap<String, Array<B>>,
}

impl<B: Backend> Recorder<B> {
    pub fn new() -> Self {
        Self {
            arrays: HashMap::new(),
        }
    }

    pub fn record(
        &mut self,
        path: String,
        shape: Box<[usize]>,
        data_type: DataType,
        allocation: Allocation<B>,
    ) -> Result<(), Error> {
        debug_assert!(
            allocation.size() >= size_for_shape(&shape, data_type),
            "declared shape for {path} does not fit the allocation",
        );
        debug_assert!(!self.arrays.contains_key(&path), "duplicate trace path {path}");

        self.arrays.insert(path, Array::new(shape, data_type, allocation)?);

        Ok(())
    }

    pub fn write(
        &self,
        output_path: &Path,
    ) -> Result<(), Error> {
        serialize_to_file(self.arrays.iter().map(|(path, array)| (path.as_str(), array)), None, output_path)?;

        Ok(())
    }
}
