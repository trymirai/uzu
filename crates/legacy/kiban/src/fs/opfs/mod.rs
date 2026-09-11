//! Rust wrappers over the browser [Origin Private File System][opfs] (OPFS).

mod directory_handle;
mod file;
mod file_handle;
mod storage;
mod writable_file_stream;

pub use directory_handle::DirectoryHandle;
pub use file::File;
pub use file_handle::FileHandle;
pub use storage::get_storage_root;
use wasm_bindgen::JsValue;
pub use writable_file_stream::{WritableFileStream, WriteCommandType, WriteParams};

pub type Error = JsValue;
pub type Result<T> = std::result::Result<T, Error>;
