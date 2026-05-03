mod backend;
mod command_buffer;
mod context;
mod dense_buffer;
mod device_capabilities;
mod error;
mod event;
mod kernel;
mod metal_extensions;

pub use backend::Metal;
pub use context::MetalContext;
pub use device_capabilities::MetalDeviceCapabilities;
pub use kernel::matmul::MatmulDispatchPath;
pub use kernel::quant_matmul::{QuantizedMatmulDispatchPath, encode_quantized_matmul_with_path};
pub use metal_extensions::{DeviceExt, DeviceGeneration};
