pub(super) fn fourcc(key: &str) -> Option<u32> {
    let bytes = key.as_bytes();
    if bytes.len() != 4 {
        return None;
    }
    Some((bytes[0] as u32) << 24 | (bytes[1] as u32) << 16 | (bytes[2] as u32) << 8 | bytes[3] as u32)
}
