use std::{any::Any, cell::RefCell, collections::HashMap, rc::Rc};

use crate::backends::metal::{
    KVCache, MTLContext, MetalArray,
    forward_pass::{
        ArrayId, ForwardPassStateTrait, HashMapId, SharedBuffers,
        traces::DecoderActivationTrace,
    },
};

/// Helper state for prediction head that maps ArrayIds to specific buffers.
/// Used for the multi-step pipeline: pooled → dense → activation → norm → logits.
pub struct PredictionHeadState {
    context: Rc<MTLContext>,
    buffers: HashMap<ArrayId, RefCell<MetalArray>>,
}

impl PredictionHeadState {
    pub fn with_buffers(
        context: Rc<MTLContext>,
        buffers: HashMap<ArrayId, RefCell<MetalArray>>,
    ) -> Self {
        Self {
            context,
            buffers,
        }
    }
}

impl ForwardPassStateTrait for PredictionHeadState {
    fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[RefCell<MetalArray>]> {
        ids.iter()
            .map(|id| {
                self.buffers
                    .get(id)
                    .unwrap_or_else(|| {
                        panic!(
                            "PredictionHeadState: ArrayId {:?} not found",
                            id
                        )
                    })
                    .clone()
            })
            .collect()
    }

    fn hashmaps(
        &self,
        _ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, RefCell<MetalArray>>]> {
        Box::new([])
    }

    fn aux_buffers_suffix_length(&self) -> usize {
        1
    }

    fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }

    fn shared_buffers(&self) -> &Rc<RefCell<SharedBuffers>> {
        panic!("PredictionHeadState does not use shared_buffers")
    }

    fn kv_cache(&self) -> Option<&Rc<RefCell<KVCache>>> {
        None
    }

    fn sampling_output(&self) -> Option<&RefCell<MetalArray>> {
        None
    }

    fn traces(&self) -> Option<&Rc<RefCell<DecoderActivationTrace>>> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
