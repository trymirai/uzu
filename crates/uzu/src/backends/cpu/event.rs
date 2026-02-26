use std::cell::RefCell;

use super::Cpu;
use crate::backends::common::Event;

impl Event for RefCell<u64> {
    type Backend = Cpu;
}
