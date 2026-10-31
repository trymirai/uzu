// TODO: gigaslop, clean up

#[cfg(feature = "webgpu")]
mod compiler;
#[cfg(feature = "webgpu")]
mod compiler_internal;
mod gpu_types;
mod reflection;
mod slang_sys_ext;
mod types;
#[cfg(feature = "webgpu")]
mod wrapper;

#[cfg(feature = "webgpu")]
pub use compiler::SlangCompiler;
#[cfg(feature = "webgpu")]
pub use compiler_internal::SlangTarget;
pub use gpu_types::gpu_type_gen;
pub use reflection::{SlangArgumentType, SlangKernel, SlangParameterType};
#[cfg(feature = "vulkan")]
pub use slang_sys_ext::SessionLoadModuleWithDiagnosticsExt;
pub use types::{Specializer, slang2rust};
