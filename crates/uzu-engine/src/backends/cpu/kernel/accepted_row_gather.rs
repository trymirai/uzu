use uzu_engine_macros::kernel;

#[kernel(AcceptedRowGather)]
pub fn accepted_row_gather(
    dst: *mut u8,
    src: *const u8,
    accepted_indices: *const u32,
    max_tokens: u32,
    dst_stride: u32,
    row_bytes: u32,
) {
    let max_tokens = max_tokens as usize;
    let dst_stride = dst_stride as usize;
    let row_bytes = row_bytes as usize;
    let accepted_len = unsafe { *accepted_indices } as usize;
    assert!(accepted_len <= max_tokens, "accepted length exceeds the dispatched row capacity");
    for token in 0..accepted_len {
        let src_row = unsafe { *accepted_indices.add(accepted_len - token) } as usize;
        for offset in 0..row_bytes {
            unsafe {
                *dst.add(token * dst_stride + offset) = *src.add(src_row * row_bytes + offset);
            }
        }
    }
}
