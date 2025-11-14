use std::{any::Any, cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    DataType,
    backends::metal::{
        KVCache, MTLContext, MetalArray,
        forward_pass::{
            ArrayId, ForwardPassStateTrait, HashMapId, SharedBuffers,
            traces::DecoderActivationTrace,
        },
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
        eprintln!(
            "[DEBUG] PredictionHeadState::new - Allocating buffer: batch_size={}, feature_dim={}, data_type={:?}, buffer_size={} bytes",
            batch_size, feature_dim, data_type, buffer_size
        );
        let mtl_buffer = context.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let main_buffer = unsafe {
            MetalArray::new(mtl_buffer, &[batch_size, feature_dim], data_type)
        };

        eprintln!(
            "[DEBUG] PredictionHeadState::new - Created MetalArray with shape [{}, {}]",
            batch_size, feature_dim
        );

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
                ArrayId::Main => {
                    // Debug: verify shape when accessed
                    use crate::Array;
                    let array_ref = self.main_buffer.borrow();
                    let shape = Array::shape(&*array_ref).to_vec();
                    let buffer_len = Array::buffer(&*array_ref).len();
                    drop(array_ref);
                    eprintln!(
                        "[DEBUG] PredictionHeadState::arrays - Returning Main: shape={:?}, buffer_len={} bytes",
                        shape, buffer_len
                    );
                    self.main_buffer.clone()
                },
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

impl ForwardPassStateTrait for PredictionHeadState {
    fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        self.arrays(ids)
    }

    fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        self.hashmaps(ids)
    }

    fn aux_buffers_suffix_length(&self) -> usize {
        1 // Prediction head operates on single pooled vector
    }

    fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }

    fn shared_buffers(&self) -> &Rc<RefCell<SharedBuffers>> {
        panic!("PredictionHeadState does not use shared_buffers")
    }

    fn kv_cache(&self) -> Option<&Rc<RefCell<KVCache>>> {
        None // Prediction head doesn't use KV cache
    }

    fn sampling_output(&self) -> Option<&ArrayCell> {
        None // Prediction head doesn't have sampling output
    }

    fn traces(&self) -> Option<&Rc<RefCell<DecoderActivationTrace>>> {
        None // Prediction head uses ClassifierActivationTrace, not DecoderActivationTrace
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
