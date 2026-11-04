use uzu_engine_macros::uzu_test;

use super::*;
use crate::{backends::cpu::Cpu, engine::Engine};

#[uzu_test]
fn selected_logp_uses_full_vocabulary_and_stays_stable_at_large_offsets() {
    let result = scores_from_logits::<Cpu>(&[2.0f32.powi(60); 2], &[1]).unwrap();
    assert!((result.log_probabilities[0] + 2f64.ln()).abs() < 1e-12);
    assert!((result.label_mass - 0.5).abs() < 1e-12);
    let result = scores_from_logits::<Cpu>(&[0.0, 1.0, 2.0], &[2, 0]).unwrap();
    let denominator = 1.0 + 1f64.exp() + 2f64.exp();
    assert!((result.log_probabilities[0] - (2.0 - denominator.ln())).abs() < 1e-12);
    assert!((result.label_mass - (1.0 + 2f64.exp()) / denominator).abs() < 1e-12);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(scores_from_logits::<Cpu>(&[0.0, value], &[0]), Err(TokenScoreError::InvalidModelOutput)));
    }
}

#[uzu_test]
#[ignore]
fn native_cpu_model_scores_match_independent_full_decoder_readout() {
    native_parity::<Cpu>();
}

#[cfg(backend = "metal")]
#[uzu_test]
#[ignore]
fn native_metal_model_scores_match_independent_full_decoder_readout() {
    native_parity::<crate::backends::metal::Metal>();
}

fn native_parity<B: Backend>()
where
    TokenScoreError<B>: std::fmt::Debug,
{
    let path = std::env::var("UZU_SCORE_MODEL").expect("set UZU_SCORE_MODEL to an existing native model directory");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let engine = Engine::<B>::new().await.unwrap();
        let model = engine.load_language_model(std::path::Path::new(&path)).await.unwrap();
        let context = &model.engine.context;
        assert!(matches!(model.score_token_ids(&[], &[0], None).await, Err(TokenScoreError::EmptyPrompt)));
        assert!(matches!(model.score_token_ids(&[1], &[], None).await, Err(TokenScoreError::EmptyLabels)));
        assert!(matches!(model.score_token_ids(&[1], &[1, 1], None).await, Err(TokenScoreError::DuplicateLabel)));
        assert!(matches!(model.score_token_ids(&[1, 1], &[1], Some(1)).await, Err(TokenScoreError::ContextOverflow)));
        let vocab = model.decoder.embedding().vocab_size();
        assert!(matches!(model.score_token_ids(&[vocab as u64], &[1], None).await, Err(TokenScoreError::InvalidToken)));
        for length in [1usize, 1023, 1024, 1025] {
            let prompt = vec![42u64; length];
            let scored = model.score_token_ids(&prompt, &[10, 20], Some(2048)).await.unwrap();
            assert!(!scored.prefill_duration.is_zero());
            let mut state = model.decoder.create_empty_state(Some(2048), context).unwrap();
            state.prepare(((length - 1) / 512 * 512) as u32, length.min(512) as u32).unwrap();
            let mut commands = context.create_command_buffer(None, None).unwrap();
            let mut row = context.create_buffer_gpu(vocab as usize * 2).unwrap();
            for (batch, part) in prompt.chunks(512).enumerate() {
                let count = part.len() as u32;
                let final_batch = (batch + 1) * 512 >= length;
                let ids = commands.allocate_constant_from_slice(&vec![42u32; part.len()]).unwrap();
                let nodes = (0..count)
                    .map(|i| TrieNode {
                        trie_start: i,
                        trie_end: count,
                        height: i,
                    })
                    .collect::<Vec<_>>();
                let output = model
                    .decoder
                    .encode(
                        &ids,
                        &BatchTopology::new(&nodes, true),
                        final_batch.then(|| (count - 1..count).into()),
                        None,
                        &mut state,
                        &mut commands,
                    )
                    .unwrap();
                if let Some(logits) = output.logits.as_ref() {
                    commands.encode_copy(logits, &mut row);
                }
                state.encode_accept(&(0..count).collect::<Vec<_>>(), &mut commands).unwrap();
                drop(output);
                drop(ids);
            }
            commands.end_encoding().submit().wait_until_completed().await.unwrap();
            let logits = context.copyout::<bf16>(&row).await.unwrap();
            let max = logits.iter().map(|x| f64::from(x.to_f32())).fold(f64::NEG_INFINITY, f64::max);
            let denominator: f64 = logits.iter().map(|x| (f64::from(x.to_f32()) - max).exp()).sum();
            let mut max_logp_error = 0.0f64;
            for (actual, id) in scored.log_probabilities.iter().zip([10usize, 20]) {
                let expected = (f64::from(logits[id].to_f32()) - max) - denominator.ln();
                max_logp_error = max_logp_error.max((actual - expected).abs());
                assert!((actual - expected).abs() < 0.1, "length={length}, actual={actual}, expected={expected}");
            }
            let expected_mass: f64 = [10usize, 20]
                .into_iter()
                .map(|id| (f64::from(logits[id].to_f32()) - max).exp() / denominator)
                .sum();
            let log_mass_error = (scored.label_mass.ln() - expected_mass.ln()).abs();
            assert!(
                log_mass_error < 0.1,
                "length={length}, mass={}, expected={expected_mass}",
                scored.label_mass
            );
            let different = model.score_token_ids(&vec![43u64; length], &[10, 20], Some(2048)).await.unwrap();
            assert_ne!(scored.log_probabilities, different.log_probabilities, "B must exercise a different score");
            let again = model.score_token_ids(&prompt, &[10, 20], Some(2048)).await.unwrap();
            assert_eq!(scored.log_probabilities, again.log_probabilities);
            assert_eq!(scored.label_mass, again.label_mass);
            println!(
                "native parity length={length}: max_logp_error={max_logp_error}, log_mass_error={log_mass_error}, A-B-A passed"
            );
        }
    });
}
