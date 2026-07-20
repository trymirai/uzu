use proc_macros::kernel;

use crate::backends::common::gpu_types::ring::RingParams;

#[kernel(ContextRingUpdate)]
pub fn context_ring_update(
    input: *const u32,
    context_ring: *mut u32,
    suffix_repetition_length: u32,
    input_length: u32,
) {
    unsafe {
        let ring = &mut *context_ring.cast::<RingParams>();
        let ring_tokens = context_ring.add(2);

        for input_index in 0..input_length {
            let slot = if ring.ring_length < suffix_repetition_length {
                let slot = (ring.ring_offset + ring.ring_length) % suffix_repetition_length;
                ring.ring_length += 1;
                slot
            } else {
                let slot = ring.ring_offset;
                ring.ring_offset = (ring.ring_offset + 1) % suffix_repetition_length;
                slot
            };
            *ring_tokens.add(slot as usize) = *input.add(input_index as usize);
        }
    }
}
