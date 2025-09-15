#[cfg(feature = "structured")]
pub mod xgrammar_support {
    use half::{bf16, f16};
    use uzu_xgrammar::Matcher;

    use crate::{Array, DataType, backends::metal::MetalArray};

    pub struct StructuredState {
        pub matchers: Vec<Matcher>,
        pub bitmask_words: Vec<u32>,
        pub nwords32: usize,
        pub vocab: usize,
        pub batch: usize,
    }

    pub fn init_structured_state(
        tokenizer_info_json: &str,
        compiled_grammar_json: &str,
        batch: usize,
        vocab: usize,
    ) -> StructuredState {
        let nwords32 = (vocab + 31) / 32;
        let mut matchers = Vec::with_capacity(batch);
        for _ in 0..batch {
            let m =
                Matcher::from_json(tokenizer_info_json, compiled_grammar_json)
                    .expect("xgrammar matcher");
            matchers.push(m);
        }
        StructuredState {
            matchers,
            bitmask_words: vec![0u32; batch * nwords32],
            nwords32,
            vocab,
            batch,
        }
    }

    pub fn fill_mask_row0(state: &mut StructuredState) {
        let row_slice = &mut state.bitmask_words[0..state.nwords32];
        let ok = state.matchers[0].fill_next_token_mask_row(
            row_slice,
            1, // batch = 1 (only row 0)
            state.nwords32,
            0, // index = 0
        );
        debug_assert!(ok, "xgrammar fill_next_token_bitmask failed for row 0");
    }

    pub fn apply_mask_cpu_row0(
        logits_row0: &mut [f32],
        vocab: usize,
        mask_row0: &[u32],
        nwords32: usize,
    ) {
        for t in 0..vocab {
            let w = mask_row0[t >> 5];
            if ((w >> (t & 31)) & 1) == 0 {
                logits_row0[t] = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply mask to Metal logits for row 0 only, supporting F16/BF16/F32
    pub fn apply_mask_row0_to_metal_array(
        logits: &mut MetalArray,
        vocab: usize,
        mask_row0: &[u32],
        nwords32: usize,
    ) {
        match logits.data_type() {
            DataType::F16 => {
                let slice = logits.as_slice_mut::<f16>().unwrap();
                for t in 0..vocab {
                    let w = mask_row0[t >> 5];
                    if ((w >> (t & 31)) & 1) == 0 {
                        slice[t] = f16::NEG_INFINITY;
                    }
                }
            },
            DataType::BF16 => {
                let slice = logits.as_slice_mut::<bf16>().unwrap();
                for t in 0..vocab {
                    let w = mask_row0[t >> 5];
                    if ((w >> (t & 31)) & 1) == 0 {
                        slice[t] = bf16::NEG_INFINITY;
                    }
                }
            },
            DataType::F32 => {
                let slice = logits.as_slice_mut::<f32>().unwrap();
                for t in 0..vocab {
                    let w = mask_row0[t >> 5];
                    if ((w >> (t & 31)) & 1) == 0 {
                        slice[t] = f32::NEG_INFINITY;
                    }
                }
            },
            other => panic!("Unsupported logits dtype for mask: {:?}", other),
        }
    }

    pub fn accept_next_token(
        state: &mut StructuredState,
        token_id: i32,
    ) {
        let ok = state.matchers[0].accept(token_id);
        debug_assert!(ok, "xgrammar accept_token rejected token for row 0");
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn apply_mask_cpu_row0_basic() {
            let vocab = 10usize;
            let nwords32 = (vocab + 31) / 32; // = 1
            let mut logits_row0: Vec<f32> =
                (0..vocab).map(|i| i as f32).collect();
            let mut mask_row0 = vec![0u32; nwords32];
            // allow tokens 2 and 7 only
            mask_row0[0] = (1u32 << 2) | (1u32 << 7);

            apply_mask_cpu_row0(&mut logits_row0, vocab, &mask_row0, nwords32);

            for t in 0..vocab {
                if t == 2 || t == 7 {
                    assert_eq!(logits_row0[t], t as f32);
                } else {
                    assert!(
                        logits_row0[t].is_infinite()
                            && logits_row0[t].is_sign_negative()
                    );
                }
            }
        }

        #[test]
        fn apply_mask_cpu_row0_multiword() {
            let vocab = 70usize; // spans 3 u32 words
            let nwords32 = (vocab + 31) / 32; // = 3
            let mut logits_row0: Vec<f32> =
                (0..vocab).map(|i| i as f32).collect();
            let mut mask_row0 = vec![0u32; nwords32];
            // allow tokens across word boundaries: 0, 31, 32, 63, 64, 69
            for &t in &[0usize, 31, 32, 63, 64, 69] {
                let wi = t >> 5;
                let bi = t & 31;
                mask_row0[wi] |= 1u32 << bi;
            }

            apply_mask_cpu_row0(&mut logits_row0, vocab, &mask_row0, nwords32);

            for t in 0..vocab {
                let allowed = matches!(t, 0 | 31 | 32 | 63 | 64 | 69);
                if allowed {
                    assert_eq!(logits_row0[t], t as f32);
                } else {
                    assert!(
                        logits_row0[t].is_infinite()
                            && logits_row0[t].is_sign_negative()
                    );
                }
            }
        }

        #[test]
        #[cfg(target_os = "macos")]
        fn apply_mask_row0_to_metal_array_f32_row0_only() {
            let device = match metal::Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let suffix = 2usize;
            let vocab = 16usize;
            let total_elems = suffix * vocab;
            let bytes = (total_elems * std::mem::size_of::<f32>()) as u64;
            let buffer = device.new_buffer(
                bytes,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let mut arr = unsafe {
                crate::backends::metal::MetalArray::new(
                    buffer,
                    &[suffix, vocab],
                    crate::DataType::F32,
                )
            };
            // fill row0: 100+idx, row1: 200+idx
            {
                let slice = arr.as_slice_mut::<f32>().unwrap();
                for i in 0..vocab {
                    slice[i] = 100.0 + i as f32;
                }
                for i in 0..vocab {
                    slice[vocab + i] = 200.0 + i as f32;
                }
            }

            let nwords32 = (vocab + 31) / 32; // = 1
            let mut mask_row0 = vec![0u32; nwords32];
            // allow tokens 3, 5, 8
            for &t in &[3usize, 5, 8] {
                mask_row0[0] |= 1u32 << t;
            }

            apply_mask_row0_to_metal_array(
                &mut arr, vocab, &mask_row0, nwords32,
            );

            let slice = arr.as_slice::<f32>().unwrap();
            for t in 0..vocab {
                let v = slice[t];
                if matches!(t, 3 | 5 | 8) {
                    assert_eq!(v, 100.0 + t as f32);
                } else {
                    assert!(v.is_infinite() && v.is_sign_negative());
                }
            }
            // Row1 unchanged
            for t in 0..vocab {
                let v = slice[vocab + t];
                assert_eq!(v, 200.0 + t as f32);
            }
        }
    }
}
