use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    DataType,
    backends::metal::{
        MTLContext, MetalArray,
        forward_pass::{ArrayId, HashMapId},
    },
};

type ArrayCell = RefCell<MetalArray>;

/// Simplified state for prediction head that works with 2D tensors [batch, features]
/// Unlike ForwardPassState which is designed for 3D sequence data [batch, seq, features]
pub struct PredictionHeadState {
    context: Rc<MTLContext>,
    /// Main buffer for [batch, features] operations
    main_buffer: ArrayCell,
}

impl PredictionHeadState {
    pub fn new(
        context: Rc<MTLContext>,
        batch_size: usize,
        feature_dim: usize,
        data_type: DataType,
    ) -> Self {
        // Allocate buffer for [batch, features]
        let buffer_size = batch_size * feature_dim * data_type.size_in_bytes();
        let mtl_buffer = context.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let main_buffer = unsafe {
            MetalArray::new(mtl_buffer, &[batch_size, feature_dim], data_type)
        };

        Self {
            context,
            main_buffer: RefCell::new(main_buffer),
        }
    }

    /// Copy input data into the main buffer
    pub fn set_input(
        &self,
        input: &MetalArray,
    ) {
        self.main_buffer.borrow_mut().copy_from_array(input);
    }

    /// Get reference to main buffer for reading output
    pub fn get_output(&self) -> ArrayCell {
        self.main_buffer.clone()
    }

    // Implement minimal interface needed by EncodableWithState
    pub fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        ids.iter()
            .map(|id| match id {
                ArrayId::Main => self.main_buffer.clone(),
                _ => panic!("PredictionHeadState only supports ArrayId::Main"),
            })
            .collect()
    }

    pub fn hashmaps(
        &self,
        _ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        // Prediction head doesn't use hashmaps
        Box::new([])
    }

    pub fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }
}
