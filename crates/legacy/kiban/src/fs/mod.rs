#[cfg(target_family = "wasm")]
mod asyn_opfs;
#[cfg(target_family = "wasm")]
mod opfs;
mod part_file;

pub mod asyn;

pub use part_file::PartFile;
