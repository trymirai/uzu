pub fn description() -> String {
    let Ok(device) = uzu::device::Device::create() else {
        return String::new();
    };
    let mem_gb = (device.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)).round() as i64;
    match device.cpu_name.as_deref() {
        Some(cpu) if !cpu.is_empty() => format!("{cpu} · {mem_gb} GB memory"),
        _ => format!("{mem_gb} GB memory"),
    }
}
