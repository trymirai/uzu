use std::{
    collections::HashMap,
    fmt::{Arguments, Write},
    path::Path,
};

use safetensors::serialize_to_file;

use super::{Array, Error};
use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

/// Collects activations of a single forward pass under lalamo's path layout.
///
/// Paths are built from a scope stack that mirrors the block hierarchy, so a
/// capture point only ever names its own field: `Transformer::encode` pushes
/// `layer_results.3`, `TransformerLayer::encode` pushes `activation_trace`, and
/// the capture itself contributes `mlp` — yielding
/// `activation_trace.layer_results.3.activation_trace.mlp`.
pub struct Recorder<B: Backend> {
    arrays: HashMap<String, Array<B>>,
    scope: String,
    scope_lengths: Vec<usize>,
}

impl<B: Backend> Recorder<B> {
    pub fn new() -> Self {
        Self {
            arrays: HashMap::new(),
            scope: String::new(),
            scope_lengths: Vec::new(),
        }
    }

    pub fn push_scope(
        &mut self,
        segment: Arguments<'_>,
    ) {
        self.scope_lengths.push(self.scope.len());
        if !self.scope.is_empty() {
            self.scope.push('.');
        }
        self.scope.write_fmt(segment).expect("writing into a String cannot fail");
    }

    pub fn pop_scope(&mut self) {
        let length = self.scope_lengths.pop().expect("popped more trace scopes than were pushed");
        self.scope.truncate(length);
    }

    /// Full path of `name` under the current scope.
    pub fn path(
        &self,
        name: &str,
    ) -> String {
        if self.scope.is_empty() {
            return name.to_owned();
        }

        format!("{}.{}", self.scope, name)
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

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn write(
        &self,
        output_path: &Path,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<(), Error> {
        serialize_to_file(self.arrays.iter().map(|(path, array)| (path.as_str(), array)), metadata, output_path)?;

        Ok(())
    }
}

impl<B: Backend> Default for Recorder<B> {
    fn default() -> Self {
        Self::new()
    }
}
