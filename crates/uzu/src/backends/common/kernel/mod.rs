use super::Backend;

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    // Every kernel should have a trait here
}
