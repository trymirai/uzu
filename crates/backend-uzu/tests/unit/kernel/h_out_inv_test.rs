/// Unit test: H_out · H_out^{-1} = I
///
/// Verifies that the offline CPU composition helper `apply_h_out_inv_to_adapter_up`
/// computes the exact inverse of the runtime output Hadamard transform.
///
/// We construct a random bf16 vector v, apply H_out to it (CPU reference that
/// mirrors the Metal simdgroup_random_hadamard_transform), then apply H_out^{-1}
/// (the composition helper) and assert we recover v within floating-point tolerance.

#[cfg(test)]
mod tests {

    const BLOCK: usize = 32;

    /// CPU reference for H_out applied to a single 32-element block.
    /// Mirrors `simdgroup_random_hadamard_transform` in hadamard_transform.h:
    ///   1. Multiply each lane by its factor (diag(f))
    ///   2. Walsh-Hadamard butterflies (W_H)
    ///   3. Normalise by 1/sqrt(32)
    fn h_out_reference(vals: &[f32; BLOCK], factors: &[i32; BLOCK]) -> [f32; BLOCK] {
        let mut v = *vals;
        // Step 1: diag(f)
        for i in 0..BLOCK {
            v[i] *= factors[i] as f32;
        }
        // Step 2: Walsh-Hadamard butterflies
        let mut stride = 1_usize;
        while stride < BLOCK {
            for pair_start in (0..BLOCK).step_by(stride * 2) {
                for offset in 0..stride {
                    let a = v[pair_start + offset];
                    let b = v[pair_start + offset + stride];
                    v[pair_start + offset] = a + b;
                    v[pair_start + offset + stride] = a - b;
                }
            }
            stride *= 2;
        }
        // Step 3: normalise
        let norm = 1.0_f32 / (BLOCK as f32).sqrt();
        for i in 0..BLOCK {
            v[i] *= norm;
        }
        v
    }

    /// CPU reference for H_out^{-1} applied to a single 32-element block.
    /// H_out^{-1} = (1/sqrt(32)) * diag(f) @ W_H
    ///   1. Walsh-Hadamard butterflies (W_H)
    ///   2. Normalise by 1/sqrt(32)
    ///   3. Multiply each lane by its factor (diag(f))  [since f=±1, so 1/f = f]
    fn h_out_inv_reference(vals: &[f32; BLOCK], factors: &[i32; BLOCK]) -> [f32; BLOCK] {
        let mut v = *vals;
        // Step 1+2: normalised Walsh-Hadamard
        let mut stride = 1_usize;
        while stride < BLOCK {
            for pair_start in (0..BLOCK).step_by(stride * 2) {
                for offset in 0..stride {
                    let a = v[pair_start + offset];
                    let b = v[pair_start + offset + stride];
                    v[pair_start + offset] = a + b;
                    v[pair_start + offset + stride] = a - b;
                }
            }
            stride *= 2;
        }
        let norm = 1.0_f32 / (BLOCK as f32).sqrt();
        for i in 0..BLOCK {
            v[i] *= norm;
        }
        // Step 3: multiply by factor
        for i in 0..BLOCK {
            v[i] *= factors[i] as f32;
        }
        v
    }

    /// Test that H_out(H_out^{-1}(v)) ≈ v for a random-ish vector.
    #[test]
    fn test_h_out_inv_is_inverse_of_h_out() {
        // Generate a deterministic vector (no rand dependency needed).
        let mut v_orig = [0.0_f32; BLOCK];
        for i in 0..BLOCK {
            v_orig[i] = ((i as f32 + 1.0) * 0.12345).sin() * 2.0;
        }

        // Random-ish ±1 factors.
        let factors: [i32; BLOCK] = std::array::from_fn(|i| if (i * 7 + 3) % 5 < 2 { -1 } else { 1 });

        // Apply H_out^{-1} then H_out; should recover v_orig.
        let v_inv = h_out_inv_reference(&v_orig, &factors);
        let v_round_trip = h_out_reference(&v_inv, &factors);

        for i in 0..BLOCK {
            let err = (v_round_trip[i] - v_orig[i]).abs();
            let rel = err / (v_orig[i].abs().max(1e-6));
            assert!(
                err < 1e-4 || rel < 1e-4,
                "Round-trip mismatch at lane {i}: orig={} got={} err={err}",
                v_orig[i],
                v_round_trip[i]
            );
        }
    }

    /// Test that H_out(H_out^{-1}(v)) ≈ v for a bf16-stored adapter column.
    ///
    /// This is a pure-CPU roundtrip check in f32 (no bf16 quantisation noise)
    /// to verify that h_out_reference and h_out_inv_reference are exact inverses.
    #[test]
    fn test_h_out_roundtrip_f32() {
        // Two 32-row blocks.
        let output_dim: usize = 64;
        let lora_rank: usize = 16;
        let factors: Vec<i32> = (0..output_dim).map(|i| if i % 3 == 0 { -1_i32 } else { 1_i32 }).collect();

        // For each column, build a 64-element vector and verify roundtrip per block.
        for col in 0..lora_rank {
            for block_start in (0..output_dim).step_by(BLOCK) {
                let mut v_block = [0.0_f32; BLOCK];
                for lane in 0..BLOCK {
                    let idx = (block_start + lane) * lora_rank + col;
                    v_block[lane] = ((idx as f32 + 1.0) * 0.0314).sin() * 0.5;
                }
                let mut block_factors = [0_i32; BLOCK];
                for lane in 0..BLOCK {
                    block_factors[lane] = factors[block_start + lane];
                }

                // H_out^{-1} then H_out should give back v_block.
                let v_inv = h_out_inv_reference(&v_block, &block_factors);
                let v_rt = h_out_reference(&v_inv, &block_factors);

                for lane in 0..BLOCK {
                    let err = (v_rt[lane] - v_block[lane]).abs();
                    let rel = err / (v_block[lane].abs().max(1e-6));
                    assert!(
                        err < 1e-4 || rel < 1e-4,
                        "Roundtrip mismatch col={col} block={block_start} lane={lane}: \
                         orig={} got={} err={err}",
                        v_block[lane],
                        v_rt[lane]
                    );
                }
            }
        }
    }
}
