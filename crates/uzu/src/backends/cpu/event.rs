use std::{pin::Pin, sync::atomic::AtomicU64};

use crate::backends::{common::Event, cpu::Cpu};

impl Event for Pin<Box<AtomicU64>> {
    type Backend = Cpu;
}
