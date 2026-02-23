use std::{cell::RefCell, sync::Arc};

use crate::backends::{common::Event, cpu::backend::Cpu};

pub type CpuEvent = Arc<RefCell<u64>>;

impl Event for CpuEvent {
    type Backend = Cpu;
}
