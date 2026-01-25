use super::Backend;

pub trait Kernels: Sized {
    type Backend: Backend;
}
