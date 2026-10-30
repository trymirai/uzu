use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub struct MakeSendFuture<F>(F);

impl<F> MakeSendFuture<F> {
    pub fn new(future: F) -> Self {
        Self(future)
    }
}

impl<F: Future> Future for MakeSendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Self::Output> {
        // The inner future remains pinned for the lifetime of this wrapper.
        unsafe { self.map_unchecked_mut(|this| &mut this.0).poll(cx) }
    }
}

#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
unsafe impl<F> Send for MakeSendFuture<F> {}
