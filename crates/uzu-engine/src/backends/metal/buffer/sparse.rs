use std::sync::Arc;

use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt,
    MTLResidencySet, MTLResourceOptions, MTLSharedEventExt, MTLSharedEventListener, MTLSharedEventNotificationBlock,
    MTLSparseTextureMappingMode,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{Buffer, SparseBuffer},
    metal::{Metal, MetalContext, error::MetalError, heaps::MetalHeapPage, metal_extensions::SparsePageSizeExt},
};

pub struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pages: Vec<MetalHeapPage>,
    length: usize,
    capacity: usize,
    context: Arc<MetalContext>,
}

impl MetalSparseBuffer {
    pub(in crate::backends::metal) fn new(
        context: Arc<MetalContext>,
        capacity: usize,
    ) -> Result<Self, MetalError> {
        let page_size = context.heaps.page_size();

        let buffer = context
            .device
            .new_buffer_with_length_options_placement_sparse_page_size(
                capacity.next_multiple_of(page_size.in_bytes()),
                MTLResourceOptions::STORAGE_MODE_PRIVATE,
                page_size,
            )
            .ok_or(MetalError::CannotCreateBuffer)?;

        let residency_set_locked = context.residency_set.lock();
        residency_set_locked.add_allocation(buffer.as_ref());
        residency_set_locked.commit();
        residency_set_locked.request_residency();
        drop(residency_set_locked);

        Ok(Self {
            buffer,
            pages: Vec::new(),
            length: 0,
            capacity,
            context,
        })
    }

    pub(super) fn mtl_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }
}

impl Buffer for MetalSparseBuffer {
    type Backend = Metal;

    fn size(&self) -> usize {
        self.capacity
    }
}

impl SparseBuffer for MetalSparseBuffer {
    fn map(
        &mut self,
        until: usize,
    ) -> Result<(), MetalError> {
        if until <= self.length {
            return Ok(());
        }

        let old_page_count = self.length.div_ceil(self.context.heaps.page_size().in_bytes());
        let new_page_count = until.div_ceil(self.context.heaps.page_size().in_bytes());

        if new_page_count > old_page_count {
            let mut sparse_state = self.context.sparse_state.lock();
            let batches = self.context.heaps.allocate(new_page_count - old_page_count)?;
            let mut current_page_count = old_page_count;
            for batch in &batches {
                let num_pages_in_batch = batch.range().iter().len();
                self.context.sparse_queue.update_buffer_mappings(
                    &self.buffer,
                    Some(batch.heap()),
                    &[MTL4UpdateSparseBufferMappingOperation::new(
                        MTLSparseTextureMappingMode::Map,
                        current_page_count..current_page_count + num_pages_in_batch,
                        batch.range().start,
                    )],
                );
                current_page_count += num_pages_in_batch;
            }
            sparse_state.did_ops = true;
            self.pages.extend(batches);
        }

        self.length = until;

        Ok(())
    }
}

impl Drop for MetalSparseBuffer {
    fn drop(&mut self) {
        if self.length > 0 {
            let mut sparse_state = self.context.sparse_state.lock();
            self.context.sparse_queue.update_buffer_mappings(
                &self.buffer,
                None,
                &[MTL4UpdateSparseBufferMappingOperation::new(
                    MTLSparseTextureMappingMode::Unmap,
                    0..self.length.div_ceil(self.context.heaps.page_size().in_bytes()),
                    0,
                )],
            );
            sparse_state.did_ops = true;
            let event = self.context.sparse_queue.device().new_shared_event().unwrap();
            let retain = (self.buffer.clone(), std::mem::take(&mut self.pages), event.clone());
            event.notify_listener_at_value(
                &MTLSharedEventListener::shared_listener(),
                1,
                &MTLSharedEventNotificationBlock::new(move |_, _| {
                    let _ = &retain;
                }),
            );
            self.context.sparse_queue.signal_event_value(event.as_ref(), 1);
        }

        let residency_set_locked = self.context.residency_set.lock();
        residency_set_locked.remove_allocation(self.buffer.as_ref());
        residency_set_locked.commit();
    }
}
