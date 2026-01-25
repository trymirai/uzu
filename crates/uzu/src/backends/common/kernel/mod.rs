use super::Backend;

pub trait Kernels: Sized {
    type Backend: Backend;

    // Every kernel should have a trait here
}
