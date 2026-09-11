#[cfg(backend = "metal")]
use std::{cell::OnceCell, sync::Arc};

#[cfg(backend = "metal")]
use crate::backends::{common::Context, metal::MetalContext};

#[cfg(backend = "metal")]
pub fn shared_metal_context() -> Arc<MetalContext> {
    thread_local! {
        static CTX: OnceCell<Arc<MetalContext>> = const { OnceCell::new() };
    }
    CTX.with(|cell| cell.get_or_init(|| MetalContext::new().expect("Metal context")).clone())
}

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}
