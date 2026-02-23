// TODO: move
pub mod argmax;

mod backend;
mod buffer;
mod command_buffer;
mod context;
mod error;
mod event;
mod kernel;

pub use backend::Cpu;
