use std::path::{Path, PathBuf};

pub fn save_file(
    title: &str,
    default_name: &str,
    extension: &str,
    type_label: &str,
) -> Option<PathBuf> {
    rfd::FileDialog::new().set_title(title).set_file_name(default_name).add_filter(type_label, &[extension]).save_file()
}

pub fn write_bytes(
    path: &Path,
    data: &[u8],
) -> bool {
    std::fs::write(path, data).is_ok()
}
