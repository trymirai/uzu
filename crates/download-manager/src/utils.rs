use crate::{Path, Uuid};

#[inline]
pub fn compute_download_id(
    url: &str,
    dest: &Path,
) -> Uuid {
    // Stable namespace for this crate; constant to ensure deterministic IDs.
    // This is a randomly generated namespace UUID v4, hardcoded.
    // Changing this will change the computed IDs across the app.
    const NAMESPACE: Uuid = Uuid::from_bytes([
        0x7d, 0xa6, 0x81, 0xb9, 0x12, 0x5f, 0x4f, 0x6b, 0x8a, 0x1f, 0x85, 0x5c, 0x84, 0x3a, 0x3c, 0x5e,
    ]);
    let key = format!("{}\n{}", url, dest.display());
    Uuid::new_v5(&NAMESPACE, key.as_bytes())
}
