use uzu_engine_macros::kernel;

#[kernel(AcceptedRowGather)]
pub fn accepted_row_gather(
    dst: *mut u8,
    src: *const u8,
    accepted_indices: *const u32,
    num_tokens: u32,
    dst_stride: u32,
    row_bytes: u32,
) {
    let num_tokens = num_tokens as usize;
    let dst_stride = dst_stride as usize;
    let row_bytes = row_bytes as usize;
    let accepted_len = unsafe { *accepted_indices } as usize;
    for token in 0..num_tokens {
        let src_row = unsafe { *accepted_indices.add(accepted_len - token) } as usize;
        for offset in 0..row_bytes {
            unsafe {
                *dst.add(token * dst_stride + offset) = *src.add(src_row * row_bytes + offset);
            }
        }
    }
}
