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

    let model_size_bytes = get_directory_size(path).unwrap_or(0);

    let mut sys = System::new();
    sys.refresh_memory();

    let allowed_bytes = sys.total_memory() * 60 / 100;
    return model_size_bytes <= allowed_bytes;
}
