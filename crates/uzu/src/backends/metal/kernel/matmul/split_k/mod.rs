mod dispatch_descriptor;
mod kernel;
mod pipeline_configuration;
mod tile_configuration;

pub use kernel::Kernel as SplitKGemm;
pub(crate) use dispatch_descriptor::DispatchDescriptor;
