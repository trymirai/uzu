// TODO: gigaslop, clean up

#[cfg(feature = "webgpu")]
mod compiler;
#[cfg(feature = "webgpu")]
mod compiler_internal;
#[cfg(feature = "webgpu")]
mod gpu_types;
#[cfg(feature = "webgpu")]
mod reflection;
mod slang_sys_ext;
#[cfg(feature = "webgpu")]
mod types;
#[cfg(feature = "webgpu")]
mod wrapper;

#[cfg(feature = "webgpu")]
pub use compiler::SlangCompiler;
#[cfg(feature = "webgpu")]
pub use compiler_internal::SlangTarget;
#[cfg(feature = "webgpu")]
pub use reflection::{SlangArgumentType, SlangBufferAccess, SlangKernel, SlangParameterType};
pub use slang_sys_ext::SessionLoadModuleWithDiagnosticsExt;
#[cfg(feature = "webgpu")]
pub use types::{Specializer, slang2rust};
