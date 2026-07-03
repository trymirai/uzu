use proc_macros::kernel;

#[kernel(TokenCopySampled)]
pub fn token_copy_sampled(
    src: *const u32,
    dst: *mut u64,
) {
    unsafe {
        let token = *src.add(0);
        *dst.add(0) = token as u64;
    }
}
