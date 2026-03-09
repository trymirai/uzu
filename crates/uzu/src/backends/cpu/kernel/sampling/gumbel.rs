use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

// Philox4x32-10 RNG (matches Metal implementation in rng.metal)

const PHILOX_W32_0: u32 = 0x9E3779B9;
const PHILOX_W32_1: u32 = 0xBB67AE85;
const PHILOX_M4X32_0: u32 = 0xD2511F53;
const PHILOX_M4X32_1: u32 = 0xCD9E8D57;

struct PhiloxState {
    key: [u32; 2],
    ctr: [u32; 4],
    state_idx: usize,
    output: [u32; 4],
}

impl PhiloxState {
    pub fn new(
        seed: u64,
        offset: u64,
    ) -> Self {
        let key = [seed as u32, (seed >> 32) as u32];
        let ctr = [offset as u32, (offset >> 32) as u32, 0, 0];

        let output = Self::curand_philox4x32_10(&ctr, &key);
        let mut state = PhiloxState {
            key,
            ctr,
            state_idx: 0,
            output,
        };
        state.incr();
        state
    }

    pub fn incr(&mut self) {
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] != 0 {
            return;
        }

        self.ctr[1] = self.ctr[1].wrapping_add(1);
        if self.ctr[1] != 0 {
            return;
        }

        self.ctr[2] = self.ctr[2].wrapping_add(1);
        if self.ctr[2] != 0 {
            return;
        }

        self.ctr[3] = self.ctr[3].wrapping_add(1);
    }

    pub fn next(&mut self) -> u32 {
        if self.state_idx >= 4 {
            self.output = Self::curand_philox4x32_10(&self.ctr, &self.key);
            self.incr();
            self.state_idx = 0;
        }
        let val = self.output[self.state_idx];
        self.state_idx += 1;
        val
    }

    fn uniform_float(&mut self) -> f32 {
        (self.next() as f32) * (1.0f32 / 4294967296.0f32)
    }

    fn mulhilo32(
        a: u32,
        b: u32,
    ) -> (u32, u32) {
        let product = (a as u64) * (b as u64);
        (product as u32, (product >> 32) as u32)
    }

    fn philox4x32round(
        ctr: &mut [u32; 4],
        key: &[u32; 2],
    ) {
        let (lo0, hi0) = Self::mulhilo32(PHILOX_M4X32_0, ctr[0]);
        let (lo1, hi1) = Self::mulhilo32(PHILOX_M4X32_1, ctr[2]);

        let new_ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
        *ctr = new_ctr;
    }

    fn philox4x32bumpkey(key: &mut [u32; 2]) {
        key[0] = key[0].wrapping_add(PHILOX_W32_0);
        key[1] = key[1].wrapping_add(PHILOX_W32_1);
    }

    fn curand_philox4x32_10(
        ctr: &[u32; 4],
        key: &[u32; 2],
    ) -> [u32; 4] {
        let mut local_ctr = *ctr;
        let mut local_key = *key;

        for _ in 0..9 {
            Self::philox4x32round(&mut local_ctr, &local_key);
            Self::philox4x32bumpkey(&mut local_key);
        }
        Self::philox4x32round(&mut local_ctr, &local_key);

        local_ctr
    }
}

#[kernel(Gumbel)]
#[variants(T, f32, f16, bf16)]
pub fn gumbel<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    batch_seeds: *const u64,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    #[specialize] in_place: bool,
) {
    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size as usize;
        let rng_seed = unsafe { *batch_seeds.add(batch_idx) };
        let mut rng = PhiloxState::new(rng_seed, 0);

        for vocab_idx in 0..vocab_size as usize {
            let global_idx = batch_start + vocab_idx;
            unsafe {
                let logit = (*logits.add(global_idx)).to_f32().unwrap();
                let u = rng.uniform_float();
                let gumbel_noise = -(-u.ln()).ln();
                *processed_logits.add(global_idx) = T::from(logit + gumbel_noise).unwrap();
            }
        }
    }
}
