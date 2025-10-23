use std::path::Path;

fn get_directory_size(path: &Path) -> std::io::Result<u64> {
    let mut size = 0u64;
    for entry_result in std::fs::read_dir(path)? {
        let entry = entry_result?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            size += get_directory_size(&entry.path())?;
        } else {
            size += metadata.len();
        }
    }
    Ok(size)
}

pub fn is_directory_fits_ram(path: &Path) -> bool {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_memory();

    let total_ram = sys.total_memory();
    let used_ram = sys.used_memory();
    let free_ram = total_ram - used_ram;

    let total_swap = sys.total_swap();
    let used_swap = sys.used_swap();
    let free_swap = total_swap - used_swap;

    let model_size_bytes = get_directory_size(path).unwrap_or(0);
    let available_total = free_ram + free_swap;
    model_size_bytes <= available_total
}
