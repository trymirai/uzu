#[cfg(metal_backend)]
use std::{cell::OnceCell, sync::Arc};

#[cfg(metal_backend)]
use crate::backends::{common::Context, metal::MetalContext};

#[cfg(metal_backend)]
pub fn shared_metal_context() -> Arc<MetalContext> {
    thread_local! {
        static CTX: OnceCell<Arc<MetalContext>> = const { OnceCell::new() };
    }
    CTX.with(|cell| cell.get_or_init(|| MetalContext::new().expect("Metal context")).clone())
}

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}
