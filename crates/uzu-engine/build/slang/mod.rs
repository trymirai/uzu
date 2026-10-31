// TODO: gigaslop, clean up

mod axis_expression;
mod compiler;
mod compiler_internal;
mod gpu_types;
mod reflection;
mod slang_gpu_types;
mod slang_sys_ext;
mod types;
#[cfg(feature = "webgpu")]
mod wrapper;

pub use axis_expression::slang_axis_expr;
pub use compiler::SlangCompiler;
pub use compiler_internal::{BlockingSlangCompiler, SlangCompilerFields, SlangTarget};
pub use gpu_types::gpu_type_gen;
pub use reflection::{SlangArgumentType, SlangKernel, SlangParameterType};
pub use slang_gpu_types::SlangGpuTypes;
pub use types::{Specializer, slang2rust};
#[cfg(feature = "webgpu")]
pub use wrapper::kernel_wrappers;
