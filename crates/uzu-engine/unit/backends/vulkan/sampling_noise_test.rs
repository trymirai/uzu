use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_test;

use super::{arg, kernel_fixture::KernelFixture};
use crate::{
    backends::vulkan::vk_kernels::{TestSamplingNoiseVulkanKernel, TestSamplingUniformVulkanKernel},
    encodable_block::sampling::{gumbel_float, revidx},
};

const SENTINEL: u32 = 0xa5a5_a5a5;
const CHUNK: u32 = 1 << 17;
const NOISE_ERROR: f32 = 1e-5;

// Independent integer oracle: each round derives its key directly; products use host u64 arithmetic.
fn philox(
    mut counter: [u32; 4],
    key: [u32; 2],
) -> [u32; 4] {
    for round in 0..10u32 {
        let k0 = key[0].wrapping_add(round.wrapping_mul(0x9e37_79b9));
        let k1 = key[1].wrapping_add(round.wrapping_mul(0xbb67_ae85));
        let p0 = u64::from(counter[0]) * 0xd251_1f53;
        let p1 = u64::from(counter[2]) * 0xcd9e_8d57;
        counter = [(p1 >> 32) as u32 ^ counter[1] ^ k0, p1 as u32, (p0 >> 32) as u32 ^ counter[3] ^ k1, p0 as u32];
    }
    counter
}

fn uniform(word: u32) -> f32 {
    (word >> 8).max(1) as f32 / 16_777_216.0
}

fn check_noise(
    value: f32,
    actual: f32,
    max_error: &mut f32,
) {
    assert!(value > 0.0 && value < 1.0);
    let cpu = -(-value.ln()).ln();
    // Each logarithm is independently evaluated in f64 and rounded to f32 before the next operation.
    let inner = -(f64::from(value).ln() as f32);
    let oracle = -(f64::from(inner).ln() as f32);
    assert!(actual.is_finite(), "uniform {value:?}: noise {actual:?}");
    for expected in [cpu, oracle] {
        let error = (actual - expected).abs();
        *max_error = max_error.max(error);
        assert!(error <= NOISE_ERROR, "uniform {value:?}: expected {expected:?}, actual {actual:?}, error {error}");
    }
}

// Frozen Philox4x32-10 known-answer vectors from Random123's tests/kat_vectors. Only the zero vector is also in
// the sampler's restricted counter layout; the other two protect the independent oracle.
#[uzu_test]
fn frozen_philox_vectors() {
    for (counter, key, expected) in [
        ([0; 4], [0; 2], [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]),
        ([u32::MAX; 4], [u32::MAX; 2], [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]),
        (
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0],
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        ),
    ] {
        assert_eq!(philox(counter, key), expected);
    }
}

#[uzu_test]
fn matches_cpu_rng_and_noise() {
    let fixture = KernelFixture::new();
    let kernel = TestSamplingNoiseVulkanKernel::new(&fixture.context).expect("TestSamplingNoise");
    let mut rows = vec![[0; 5]];
    for seed in [0, 1, 1 << 32, u64::MAX, 0x1234_5678_9abc_def0] {
        for vocab in [0, 1, 4095, 4096, 4097, 32_000, 128_256, u32::MAX] {
            for token in [0, 1, 1023, 1024, 2048, 3072, 4095, 4096, 4097, u32::MAX] {
                for word in [0, 255, 256, 0x7fff_ffff, 0xffff_ff00, u32::MAX] {
                    rows.push([seed as u32, (seed >> 32) as u32, token, vocab, word]);
                }
            }
        }
    }
    let mut rng = SmallRng::seed_from_u64(0x051d_9a23);
    rows.extend((0..1 << 16).map(|_| [0; 5].map(|_| rng.random())));
    let mut max_error = 0.0f32;
    for rows in rows.chunks(CHUNK as usize).chain(std::iter::once([].as_slice())) {
        let input = fixture.guarded(rows.as_flattened(), SENTINEL);
        let output = fixture.guarded(&vec![SENTINEL; rows.len() * 10], SENTINEL);
        let mut encoding = fixture.encoding();
        // SAFETY: distinct aligned spans hold exactly 5 input and 10 output words per row, including empty spans.
        unsafe { kernel.encode(arg(&input), arg(&output), rows.len() as u32, &mut encoding) };
        KernelFixture::complete(encoding);
        // SAFETY: the only submission touching either span has completed.
        let actual = unsafe {
            KernelFixture::assert_unchanged(&input, SENTINEL, rows.as_flattened(), "sampling inputs");
            KernelFixture::read_guarded::<u32>(&output, SENTINEL)
        };
        assert_eq!(actual.len(), rows.len() * 10);
        for (&[lo, hi, token, vocab, raw], result) in rows.iter().zip(actual.as_chunks::<10>().0) {
            let (offset, word) = revidx(token, vocab);
            assert_eq!(result[..2], [offset, word], "token {token}, vocab {vocab}");
            let words = philox([offset, 0, 0, 0], [lo, hi]);
            assert_eq!(result[2..6], words, "seed [{lo:#x}, {hi:#x}], offset {offset}");
            let value = uniform(words[word as usize]);
            assert_eq!(result[6], value.to_bits());
            let noise = f32::from_bits(result[7]);
            let cpu = gumbel_float(u64::from(lo) | u64::from(hi) << 32, (offset, word));
            assert!((noise - cpu).abs() <= NOISE_ERROR, "actual CPU seed [{lo:#x}, {hi:#x}], token {token}");
            check_noise(value, noise, &mut max_error);
            assert_eq!(result[8], uniform(raw).to_bits());
            check_noise(uniform(raw), f32::from_bits(result[9]), &mut max_error);
        }
    }
    eprintln!("sampling CPU/oracle maximum absolute error {max_error:e}");
    fixture.assert_clean();
}

#[uzu_test]
fn every_24_bit_uniform() {
    let fixture = KernelFixture::new();
    let kernel = TestSamplingUniformVulkanKernel::new(&fixture.context).expect("TestSamplingUniform");
    let mut max_error = 0.0f32;
    let mut previous = f32::NEG_INFINITY;
    for start in (0..1 << 24).step_by(CHUNK as usize) {
        let output = fixture.guarded(&vec![SENTINEL; CHUNK as usize * 2], SENTINEL);
        let mut encoding = fixture.encoding();
        // SAFETY: the nonaliasing output holds 2 words per invocation; start + count remains in the 24-bit range.
        unsafe { kernel.encode(arg(&output), start, CHUNK, &mut encoding) };
        KernelFixture::complete(encoding);
        // SAFETY: the sole submission has completed.
        let actual = unsafe { KernelFixture::read_guarded::<u32>(&output, SENTINEL) };
        assert_eq!(actual.len(), CHUNK as usize * 2);
        for (index, result) in actual.as_chunks::<2>().0.iter().enumerate() {
            let value = (start + index as u32).max(1) as f32 / 16_777_216.0;
            assert_eq!(result[0], value.to_bits());
            let noise = f32::from_bits(result[1]);
            check_noise(value, noise, &mut max_error);
            // Noise must preserve monotonicity of the discrete uniform distribution, allowing rounded equal values.
            assert!(noise >= previous, "uniform {value:?}: previous {previous:?}, noise {noise:?}");
            previous = noise;
        }
    }
    eprintln!("sampling exhaustive 16777216 inputs: maximum absolute error {max_error:e}");
    fixture.assert_clean();
}
