use std::cell::RefCell;

use crate::backends::metal::MetalArray;

pub type ArrayCell = RefCell<MetalArray>;
