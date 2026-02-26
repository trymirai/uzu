use dsl::kernel;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TokenCopySampled)]
pub fn token_copy_sampled(
    #[allow(unused)] src: *const u32,
    #[allow(unused)] dst: *mut u64,
) {
    todo!()
}

#[kernel(TokenCopyToResults)]
pub fn token_copy_to_results(
    #[allow(unused)] src: *const u32,
    #[allow(unused)] dst: *mut u32,
) {
    todo!()
}
