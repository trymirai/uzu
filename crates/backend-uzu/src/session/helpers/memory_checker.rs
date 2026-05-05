use std::{env, path::Path};

use crate::utils::memory::{get_free_ram, get_free_swap};

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
    if env::var_os("UZU_SKIP_MODEL_MEMORY_CHECK").is_some() {
        return true;
    }

    let model_size_bytes = get_directory_size(path).unwrap_or(0);
    let available_total = get_free_ram() + get_free_swap();
    model_size_bytes <= available_total
}
