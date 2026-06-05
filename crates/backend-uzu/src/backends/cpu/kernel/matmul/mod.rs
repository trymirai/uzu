pub(super) mod gemv;
mod kernel;
mod reference;

pub use kernel::MatmulCpuKernel;
