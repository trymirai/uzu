use std::{any::Any, cell::RefCell, collections::HashMap, rc::Rc};

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    kv_cache::KVCache,
    state::{ArrayId, HashMapId, SharedBuffers},
    traces::DecoderActivationTrace,
};
use crate::{
    Array,
    backends::metal::{MTLContext, MetalArray},
};

type ArrayCell = RefCell<MetalArray>;

/// Unified trait for forward pass state management.
///
/// This trait provides a common interface for both LLM (autoregressive) and
/// classifier (bidirectional) forward pass states, allowing the same encodables
/// to work with either mode.
///
/// LLM-specific features (KV cache, sampling, traces) are optional and return
/// None for classifier implementations.
pub trait ForwardPassStateTrait {
    /// Access arrays by their IDs.
    /// Returns borrowed array cells that can be accessed for encoding operations.
    fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]>;

    /// Access hashmaps (e.g., attention bias) by their IDs.
    fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]>;

    /// Get the suffix length (number of tokens in current batch).
    fn aux_buffers_suffix_length(&self) -> usize;

    /// Get the Metal context for device operations.
    fn mtl_context(&self) -> &Rc<MTLContext>;

    /// Get shared buffers (embeddings, RoPE) used across all layers.
    fn shared_buffers(&self) -> &Rc<RefCell<SharedBuffers>>;

    /// Get KV cache (LLM only - returns None for classifiers).
    fn kv_cache(&self) -> Option<&Rc<RefCell<KVCache>>>;

    /// Get sampling output buffer (LLM only - returns None for classifiers).
    fn sampling_output(&self) -> Option<&ArrayCell>;

    /// Get activation traces (LLM only when tracing enabled - returns None for classifiers).
    fn traces(&self) -> Option<&Rc<RefCell<DecoderActivationTrace>>>;

    /// Copy array from source to destination (optional - used by some kernels).
    fn copy_array(
        &self,
        source_array_id: ArrayId,
        destination_array: RefCell<MetalArray>,
    ) {
        destination_array
            .borrow_mut()
            .copy_from_array(&self.arrays(&[source_array_id])[0].borrow());
    }

    /// Encode a GPU-to-GPU copy from `source_array_id` into `destination_array`
    /// on the provided command buffer using a Blit encoder.
    ///
    /// Prefer this over `copy_array` when encoding within a command buffer to ensure
    /// correct GPU synchronization and avoid reading stale data from shared buffers.
    fn encode_copy_array(
        &self,
        command_buffer: &MPSCommandBuffer,
        source_array_id: ArrayId,
        destination_array: RefCell<MetalArray>,
    ) {
        // Borrow arrays to access underlying MTLBuffers and sizes
        let source_ref = self.arrays(&[source_array_id])[0].clone();
        let mut src_borrow = source_ref.borrow_mut();
        let mut dst_borrow = destination_array.borrow_mut();

        let src_buf = unsafe { src_borrow.mtl_buffer().clone() };
        let dst_buf = unsafe { dst_borrow.mtl_buffer().clone() };

        let copy_size_bytes = dst_borrow.size_in_bytes() as u64;
        debug_assert_eq!(
            dst_borrow.size_in_bytes(),
            src_borrow.size_in_bytes()
        );

        // Encode blit copy on the root Metal command buffer
        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let root = &*mtl_command_buffer;
        let blit_encoder = root.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &src_buf,
            0,
            &dst_buf,
            0,
            copy_size_bytes,
        );
        blit_encoder.end_encoding();
    }

    /// Get mutable reference to sampling method (LLM only).
    fn sampling_method_mut(
        &mut self
    ) -> Option<&mut Option<crate::session::parameter::SamplingMethod>> {
        None
    }

    /// Get sampling method (LLM only).
    fn sampling_method(
        &self
    ) -> Option<crate::session::parameter::SamplingMethod> {
        None
    }

    /// Downcast to concrete type for type-specific operations.
    fn as_any(&self) -> &dyn Any;
}
