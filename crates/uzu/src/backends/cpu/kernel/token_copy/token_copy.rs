use dsl::kernel;

#[kernel(TokenCopySampled)]
pub fn token_copy_sampled(
    src: *const u32,
    dst: *mut u64,
) {
    unsafe {
        *dst.add(0) = *src.add(0) as u64;
    }
}

#[kernel(TokenCopyToResults)]
pub fn token_copy_to_results(
    src: *const u32,
    dst: *mut u32,
) {
    unsafe {
        *dst.add(0) = *src.add(0);
    }
}
