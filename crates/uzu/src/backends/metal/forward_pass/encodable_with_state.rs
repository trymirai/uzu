use std::{cell::RefCell, collections::HashMap, rc::Rc};

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{ArrayId, HashMapId};
use crate::backends::metal::{MTLContext, MetalArray};

type ArrayCell = RefCell<MetalArray>;

pub struct EncodingParameters {
    pub warmup: bool,
    pub enable_commit: bool,
    pub wait_until_completed: bool,
    pub projection_step: Option<usize>,
}

impl EncodingParameters {
    pub fn new(
        warmup: bool,
        enable_commit: bool,
        wait_until_completed: bool,
    ) -> Self {
        Self {
            warmup,
            enable_commit,
            wait_until_completed,
            projection_step: None,
        }
    }

    pub fn with_projection(
        mut self,
        projection_step: usize,
    ) -> Self {
        self.projection_step = Some(projection_step);
        self
    }
}

/// Common interface that all forward pass state types must implement
pub trait ForwardPassStateInterface {
    fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]>;
    fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]>;
    fn aux_buffers_suffix_length(&self) -> usize;
    fn mtl_context(&self) -> &Rc<MTLContext>;
}

pub trait EncodableWithState<S: ForwardPassStateInterface + ?Sized> {
    fn encode(
        &self,
        state: &mut S,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    );
}
