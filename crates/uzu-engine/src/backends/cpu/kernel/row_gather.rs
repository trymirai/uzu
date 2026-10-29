use uzu_engine_macros::kernel;

#[kernel(RowGather)]
pub fn row_gather(
    dst: *mut u8,
    src: *const u8,
    accepted_indices: *const u32,
    row_bytes: u32,
) {
    let row_bytes = row_bytes as usize;
    let index = unsafe { *accepted_indices.add(1) } as usize;
    for offset in 0..row_bytes {
        unsafe {
            *dst.add(offset) = *src.add(index * row_bytes + offset);
        }
    }
}
