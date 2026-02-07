use std::{os::raw::c_void, ptr::NonNull};

use super::Backend;

pub trait NativeBuffer: Send + Sync + Clone {
    type Backend: Backend;

    fn set_label(
        &self,
        label: Option<&str>,
    );

    fn cpu_ptr(&self) -> NonNull<c_void>;
    fn length(&self) -> usize;
    fn id(&self) -> usize;
}
