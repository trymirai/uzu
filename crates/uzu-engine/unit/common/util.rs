#[cfg(backend = "metal")]
use std::{cell::OnceCell, sync::Arc};

#[cfg(backend = "metal")]
use crate::backends::{
    common::{Backend, Context},
    metal::Metal,
};

#[cfg(backend = "metal")]
pub fn shared_metal_context() -> Arc<<Metal as Backend>::Context> {
    thread_local! {
        static CTX: OnceCell<Arc<<Metal as Backend>::Context>> = const { OnceCell::new() };
    }
    CTX.with(|cell| cell.get_or_init(|| <Metal as Backend>::Context::new().expect("Metal context")).clone())
}

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}
