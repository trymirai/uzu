mod input_processor;
mod memory_checker;
pub use input_processor::{InputProcessor, InputProcessorDefault};
pub use memory_checker::is_directory_fits_ram;
