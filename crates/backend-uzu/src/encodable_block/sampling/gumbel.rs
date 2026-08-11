const PHILOX_W32_0: u32 = 0x9E3779B9;
const PHILOX_W32_1: u32 = 0xBB67AE85;
const PHILOX_M4X32_0: u32 = 0xD2511F53;
const PHILOX_M4X32_1: u32 = 0xCD9E8D57;

#[inline]
fn mulhilo32(
    a: u32,
    b: u32,
) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

#[inline]
fn philox4x32_round(
    ctr: &mut [u32; 4],
    key: &[u32; 2],
) {
    let (hi0, lo0) = mulhilo32(PHILOX_M4X32_0, ctr[0]);
    let (hi1, lo1) = mulhilo32(PHILOX_M4X32_1, ctr[2]);

    let new_ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];

    *ctr = new_ctr;
}

#[inline]
fn philox4x32_bumpkey(key: &mut [u32; 2]) {
    key[0] = key[0].wrapping_add(PHILOX_W32_0);
    key[1] = key[1].wrapping_add(PHILOX_W32_1);
}

fn uniform_float(
    key: u64,
    (offset, word): (u32, u32),
) -> f32 {
    let mut ctr = [offset, 0, 0, 0];

    let mut key = [key as u32, (key >> 32) as u32];

    philox4x32_round(&mut ctr, &key);

    for _ in 0..9 {
        philox4x32_bumpkey(&mut key);
        philox4x32_round(&mut ctr, &key);
    }

    unit_interval(ctr[word as usize])
}

/// Uniform in [2^-24, 1-2^-24]; mirrors Metal `uniform_float` in kernel/rng.h. -ln(-ln u)
/// needs (0,1): u == 1 gives +inf. 24 bits only; the full 32 rounds the top words to 2^32.
#[inline]
fn unit_interval(word: u32) -> f32 {
    ((word >> 8).max(1)) as f32 * (1.0 / 16777216.0)
}

pub fn gumbel_float(
    key: u64,
    (offset, word): (u32, u32),
) -> f32 {
    -f32::ln(-f32::ln(uniform_float(key, (offset, word))))
}

const THREADGROUP_SIZE: u32 = 1024;
const WORDS_PER_OFFSET: u32 = 4;

pub fn revidx(
    logit_idx: u32,
    vocab_size: u32,
) -> (u32, u32) {
    let thread_idx = logit_idx % THREADGROUP_SIZE;
    let thread_offset = vocab_size.div_ceil(THREADGROUP_SIZE * WORDS_PER_OFFSET) * thread_idx;

    let block_idx = logit_idx / THREADGROUP_SIZE;
    let block_idx_offset = block_idx / WORDS_PER_OFFSET;
    let block_idx_word = block_idx % WORDS_PER_OFFSET;

    (thread_offset + block_idx_offset, block_idx_word)
}

#[cfg(test)]
#[path = "../../../tests/unit/encodable_block/sampling/gumbel_test.rs"]
mod tests;
