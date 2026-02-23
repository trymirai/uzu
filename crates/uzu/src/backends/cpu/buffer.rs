use std::{
    os::raw::c_void,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use crate::backends::{common::NativeBuffer, cpu::backend::Cpu};

#[derive(Debug, Clone)]
pub struct CpuBuffer {
    buffer: Vec<u8>,
    id: usize,
}

impl CpuBuffer {
    pub fn new(size: usize) -> Self {
        let mut latest_id = get_counter().lock().unwrap();
        *latest_id += 1;
        Self {
            buffer: vec![0; size],
            id: *latest_id,
        }
    }

    pub fn data(&self) -> &Vec<u8> {
        &self.buffer
    }

    pub fn data_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }
}

impl NativeBuffer for CpuBuffer {
    type Backend = Cpu;

    fn set_label(
        &self,
        _label: Option<&str>,
    ) {
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        let mut_ptr = self.buffer.as_ptr() as *mut c_void;
        NonNull::new(mut_ptr).unwrap()
    }

    fn length(&self) -> usize {
        self.buffer.len()
    }

    fn id(&self) -> usize {
        self.id
    }
}

fn get_counter() -> &'static Mutex<usize> {
    static VALUE: OnceLock<Mutex<usize>> = OnceLock::new();
    VALUE.get_or_init(|| Mutex::new(0))
}
