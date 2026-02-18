use std::mem;

use mach2::{
    kern_return::KERN_SUCCESS,
    mach_types::task_t,
    message::mach_msg_type_number_t,
    task::task_info,
    task_info::{TASK_BASIC_INFO, task_basic_info},
    traps::mach_task_self,
};

pub fn get_memory_usage() -> Option<u64> {
    unsafe {
        let task: task_t = mach_task_self();

        let mut info = task_basic_info {
            virtual_size: 0,
            resident_size: 0,
            user_time: mem::zeroed(),
            system_time: mem::zeroed(),
            policy: 0,
            suspend_count: 0,
        };

        let mut count: mach_msg_type_number_t = (mem::size_of::<task_basic_info>() / mem::size_of::<u32>()) as u32;

        let result = task_info(task, TASK_BASIC_INFO, &mut info as *mut _ as *mut i32, &mut count);

        if result == KERN_SUCCESS {
            Some(info.resident_size as u64)
        } else {
            None
        }
    }
}
