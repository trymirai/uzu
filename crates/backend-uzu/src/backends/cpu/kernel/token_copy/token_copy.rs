use proc_macros::kernel;

use crate::backends::common::gpu_types::ring::RingParams;

#[kernel(TokenCopySampled)]
pub fn token_copy_sampled(
    #[allow(unused)] src: *const u32,
    #[allow(unused)] dst: *mut u64,
    #[optional(has_context_ring)] context_ring: Option<*mut u32>,
    #[optional(has_context_ring)] suffix_repetition_length: Option<u32>,
    #[specialize] has_context_ring: bool,
) {
    unsafe {
        let token = *src.add(0);
        *dst.add(0) = token as u64;

        if has_context_ring {
            let suffix_repetition_length = suffix_repetition_length.unwrap();

            let context_ring = context_ring.unwrap();
            let ring = &mut *context_ring.cast::<RingParams>();
            let ring_tokens = context_ring.add(2);
            let slot = if ring.ring_length < suffix_repetition_length {
                let slot = (ring.ring_offset + ring.ring_length) % suffix_repetition_length;
                ring.ring_length += 1;
                slot
            } else {
                let slot = ring.ring_offset;
                ring.ring_offset = (ring.ring_offset + 1) % suffix_repetition_length;
                slot
            };
            *ring_tokens.add(slot as usize) = token;
        }
    }
}

#[kernel(TokenCopyToResults)]
pub fn token_copy_to_results(
    #[allow(unused)] src: *const u32,
    #[allow(unused)] dst: *mut u32,
) {
    unsafe {
        *dst.add(0) = *src.add(0);
    }
}
