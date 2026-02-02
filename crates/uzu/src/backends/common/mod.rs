mod allocator;
mod backend;
mod buffer;
mod buffer_lifetime;
mod context;
pub mod kernel {
    include!(concat!(env!("OUT_DIR"), "/traits.rs"));
}
mod native_buffer;

pub use allocator::{AllocError, Allocator};
pub use backend::Backend;
pub use buffer::Buffer;
pub use buffer_lifetime::BufferLifetime;
pub use context::Context;
pub use kernel::Kernels;
pub use native_buffer::NativeBuffer;
