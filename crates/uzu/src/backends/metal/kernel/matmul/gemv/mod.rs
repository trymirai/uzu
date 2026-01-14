mod dispatch_descriptor;
mod kernel;
mod pipeline_configuration;

pub use kernel::Kernel as GemvKernel;
pub(crate) use dispatch_descriptor::DispatchDescriptor;
