use crate::backends::{common::Event, cpu::backend::Cpu};

pub struct CpuEvent;

impl Event for CpuEvent {
    type Backend = Cpu;
}
