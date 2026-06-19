use crate::units::Percent;

const STATES: usize = libc::CPU_STATE_MAX as usize;

pub struct CpuLoad {
    last: Vec<[u64; STATES]>,
}

impl CpuLoad {
    pub fn new() -> Self {
        Self {
            last: Vec::new(),
        }
    }

    pub fn sample(&mut self) -> Vec<Percent> {
        let Some(ticks) = read_ticks() else {
            return Vec::new();
        };
        if ticks.len() != self.last.len() {
            self.last = ticks;
            return vec![Percent(0.0); self.last.len()];
        }
        let usage = ticks
            .iter()
            .zip(&self.last)
            .map(|(current, previous)| {
                let delta = |state: usize| current[state].saturating_sub(previous[state]) as f64;
                let active = delta(libc::CPU_STATE_USER as usize)
                    + delta(libc::CPU_STATE_SYSTEM as usize)
                    + delta(libc::CPU_STATE_NICE as usize);
                let total = active + delta(libc::CPU_STATE_IDLE as usize);
                let percent = if total > 0.0 {
                    (active / total * 100.0) as f32
                } else {
                    0.0
                };
                Percent(percent.clamp(0.0, 100.0))
            })
            .collect();
        self.last = ticks;
        usage
    }
}

#[allow(deprecated)]
fn read_ticks() -> Option<Vec<[u64; STATES]>> {
    let mut cpu_count: libc::natural_t = 0;
    let mut info: libc::processor_info_array_t = core::ptr::null_mut();
    let mut info_count: libc::mach_msg_type_number_t = 0;
    let result = unsafe {
        libc::host_processor_info(
            libc::mach_host_self(),
            libc::PROCESSOR_CPU_LOAD_INFO,
            &mut cpu_count,
            &mut info,
            &mut info_count,
        )
    };
    if result != libc::KERN_SUCCESS || info.is_null() {
        return None;
    }

    let loads =
        unsafe { core::slice::from_raw_parts(info as *const libc::processor_cpu_load_info, cpu_count as usize) };
    let ticks = loads
        .iter()
        .map(|load| {
            [
                load.cpu_ticks[libc::CPU_STATE_USER as usize] as u64,
                load.cpu_ticks[libc::CPU_STATE_SYSTEM as usize] as u64,
                load.cpu_ticks[libc::CPU_STATE_IDLE as usize] as u64,
                load.cpu_ticks[libc::CPU_STATE_NICE as usize] as u64,
            ]
        })
        .collect();

    unsafe {
        libc::vm_deallocate(
            libc::mach_task_self(),
            info as libc::vm_address_t,
            info_count as usize * core::mem::size_of::<libc::c_int>(),
        );
    }
    Some(ticks)
}
