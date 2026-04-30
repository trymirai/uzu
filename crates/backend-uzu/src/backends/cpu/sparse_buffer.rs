use std::{cell::UnsafeCell, pin::Pin};

use crate::backends::{common::SparseBuffer, cpu::Cpu};

impl SparseBuffer for UnsafeCell<Pin<Box<[u8]>>> {
    type Backend = Cpu;
}
