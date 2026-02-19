mod dispatch_descriptor;
mod kernel;
mod pipeline_configuration;

pub(crate) use dispatch_descriptor::DispatchDescriptor;
pub use kernel::Kernel as GemmKernel;
