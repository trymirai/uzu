use std::{mem::ManuallyDrop, rc::Weak};

use super::{Backend, Context};

pub struct Buffer<B: Backend> {
    inner: ManuallyDrop<B::NativeBuffer>,
    context: Weak<B::Context>,
}

impl<B: Backend> Buffer<B> {
    pub fn new(
        inner: B::NativeBuffer,
        context: Weak<B::Context>,
    ) -> Self {
        Self {
            inner: ManuallyDrop::new(inner),
            context,
        }
    }

    pub fn inner(&self) -> &B::NativeBuffer {
        &self.inner
    }
}

impl<B: Backend> Drop for Buffer<B> {
    fn drop(&mut self) {
        // Safety: drop is only called once, inner is valid
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        if let Some(ctx) = self.context.upgrade() {
            ctx.allocator().handle_buffer_drop(inner);
        }
        // If context is gone, inner drops naturally here
    }
}
