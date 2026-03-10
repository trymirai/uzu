use std::{fmt::Debug, os::raw::c_void, ptr::NonNull};

use super::Backend;

pub trait Buffer: Debug {
    type Backend: Backend<Buffer = Self>;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );

    fn cpu_ptr(&self) -> NonNull<c_void>;

    fn length(&self) -> usize;

    fn id(&self) -> usize;
}
