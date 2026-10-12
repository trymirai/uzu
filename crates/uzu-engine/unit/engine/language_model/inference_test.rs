use std::{fs::File, io::BufReader, path::Path};

use half::bf16;
use tokenizers::Tokenizer;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, CommandBufferEncoding, Context},
        cpu::Cpu,
        vulkan::Vulkan,
    },
    config::model::language_model::LanguageModelConfig,
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::Decoder, sampling::PRng},
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState, stream::SamplingMethod},
    },
    parameters::ParameterLoader,
    tests::{
        helpers::{buffer_prefix_to_vec, buffer_readback, create_context, submit_command_buffer},
        path::get_test_model_path,
    },
    trie::TrieNode,
};

const PROMPTS: [&str; 3] = ["The capital of France is", "1, 2, 3, 4,", "A language model"];
const CONTINUATION: &str = "Continue.";
const MAX_GENERATED_TOKENS: usize = 4;
// Bounds the model test KV allocation.
const CONTEXT_LENGTH: u32 = 16384;

fn load_config(model_path: &Path) -> LanguageModelConfig {
    serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json")).unwrap())).unwrap()
}

fn encode<B: Backend>(
    model: &LanguageModel<B>,
    text: &str,
) -> Vec<u64> {
    model.tokenizer().encode(text, false).unwrap().get_ids().iter().map(|&id| u64::from(id)).collect()
}

fn greedy_tokens<B: Backend>(
    model: &LanguageModel<B>,
    input: &[u64],
    state: &mut LanguageModelState<B>,
) -> Vec<u64> {
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut tokens = Vec::new();
    for token in model.stream(input, state, options).unwrap_or_else(|error| panic!("{error}")) {
        let token = token.unwrap_or_else(|error| panic!("{error}"));
        tokens.push(token);
        if tokens.len() == MAX_GENERATED_TOKENS || model.generation_config().stop_token_ids.contains(&token) {
            break;
        }
    }
    tokens
}

fn run_prompts<B: Backend>(model_path: &Path) -> Vec<[Vec<u64>; 2]> {
    let model = Engine::<B>::new().unwrap_or_else(|error| panic!("{error}")).load_language_model(model_path).unwrap();
    let continuation_ids = encode(&model, CONTINUATION);
    PROMPTS
        .iter()
        .map(|prompt| {
            let prompt_ids = encode(&model, prompt);
            let mut state = model.create_empty_state(model.recommended_context_length(), 0).unwrap();
            let initial = greedy_tokens(&model, &prompt_ids, &mut state);
            let continuation = greedy_tokens(&model, &continuation_ids, &mut state);
            [initial, continuation]
        })
        .collect()
}

/// Per prompt, `(input token batch, all-row logits)` for prefill, 3 decode steps, continuation and 2 decode steps.
/// Without `forced`, each decode token is the lowest-index argmax of the previous last row; otherwise it is replayed.
fn teacher_forced<B: Backend>(
    model_path: &Path,
    forced: Option<&[Vec<(Vec<u32>, Vec<f32>)>]>,
) -> Vec<Vec<(Vec<u32>, Vec<f32>)>> {
    let config = load_config(model_path);
    let vocab_size = config.decoder_config.vocab_size as usize;
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();
    let tokenize = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();

    let context = create_context::<B>();
    let weights_file = File::open(model_path.join("model.safetensors")).unwrap();
    let weight_loader = ParameterLoader::new(&weights_file, &*context).unwrap();
    let decoder_tree = weight_loader.tree().subtree("decoder");
    let decoder = Decoder::new(&*context, &config.decoder_config, &decoder_tree, DataType::BF16).unwrap();
    weight_loader.tree().assert_all_tensors_validated().unwrap();
    let max_context_length = Some(decoder.max_context_length().unwrap_or(CONTEXT_LENGTH).min(CONTEXT_LENGTH));

    PROMPTS
        .iter()
        .enumerate()
        .map(|(prompt_index, prompt)| {
            let mut state = decoder.create_empty_state(max_context_length, &*context).unwrap();
            let schedule = [Some(tokenize(prompt)), None, None, None, Some(tokenize(CONTINUATION)), None, None];
            let mut steps: Vec<(Vec<u32>, Vec<f32>)> = Vec::new();
            for (step, input) in schedule.into_iter().enumerate() {
                let input = input.unwrap_or_else(|| match forced {
                    Some(forced) => forced[prompt_index][step].0.clone(),
                    None => {
                        let logits = &steps.last().unwrap().1;
                        let last_row = logits[logits.len() - vocab_size..].iter().enumerate();
                        let (token, _) = last_row.fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
                            if value > best.1 {
                                (index, value)
                            } else {
                                best
                            }
                        });
                        vec![token as u32]
                    },
                });
                let rows = input.len() as u32;
                state.prepare(state.context_length(), rows).unwrap();
                let mut command_buffer = context
                    .create_command_buffer(Some("teacher forcing"), Some(context.create_allocation_pool()))
                    .unwrap();
                let token_ids = command_buffer.allocate_constant_from_slice(&input[..]).unwrap();
                let trie_tokens = input.iter().map(|&id| u64::from(id)).collect::<Box<[u64]>>();
                let trie = TrieNode::flat(state.context_length() as usize, &trie_tokens, &PRng::new(0));
                let nodes = trie.linearize().token_subtrie_ranges().collect::<Box<[_]>>();
                let batch_dim = BatchTopology::new(&nodes, true);
                let output = decoder
                    .encode(&token_ids, &batch_dim, Some((0..rows).into()), None, &mut state, &mut command_buffer)
                    .unwrap();
                state.encode_accept(&(0..rows).collect::<Box<[u32]>>(), &mut command_buffer).unwrap();
                drop(token_ids);
                submit_command_buffer(command_buffer);
                let readback = buffer_readback(&*context, &output.logits.unwrap());
                let logits = buffer_prefix_to_vec::<B, bf16>(&readback, rows as usize * vocab_size)
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<f32>>();
                assert_eq!(logits.len(), rows as usize * vocab_size, "prompt {prompt:?} step {step} logits shape");
                assert!(logits.iter().all(|value| value.is_finite()), "prompt {prompt:?} step {step} non-finite");
                steps.push((input, logits));
            }
            steps
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f64::NEG_INFINITY, |max, &value| max.max(f64::from(value)));
    let exp = logits.iter().map(|&value| (f64::from(value) - max).exp()).collect::<Vec<f64>>();
    let sum = exp.iter().sum::<f64>();
    exp.into_iter().map(|value| value / sum).collect()
}

#[uzu_test]
#[ignore]
fn test_full_model_cpu_vulkan_logits() {
    // Regression limits for teacher-forced BF16 logits; these are not rounding bounds.
    let max_error_limit = 4.0 * f64::from(bf16::EPSILON.to_f32());
    let relative_l2_limit = 0.05;
    let total_variation_limit = 0.05;
    let model_path = get_test_model_path();
    let cpu = teacher_forced::<Cpu>(&model_path, None);
    let vulkan = teacher_forced::<Vulkan>(&model_path, Some(&cpu));
    let replay = teacher_forced::<Vulkan>(&model_path, Some(&cpu));

    assert_eq!(vulkan.len(), replay.len());
    for (prompt, (first, second)) in PROMPTS.iter().zip(vulkan.iter().zip(&replay)) {
        assert_eq!(first.len(), second.len(), "prompt {prompt:?} replay step count");
        for (step, ((_, first_logits), (_, second_logits))) in first.iter().zip(second).enumerate() {
            assert_eq!(first_logits.len(), second_logits.len(), "prompt {prompt:?} step {step} replay logits shape");
            let mismatch = first_logits.iter().zip(second_logits).position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(mismatch, None, "prompt {prompt:?} step {step} Vulkan replay flat logit index");
        }
    }

    assert_eq!(cpu.len(), vulkan.len());
    for (prompt, (cpu, vulkan)) in PROMPTS.iter().zip(cpu.iter().zip(&vulkan)) {
        assert_eq!(cpu.len(), vulkan.len(), "prompt {prompt:?} step count");
        for (step, ((cpu_input, cpu_logits), (_, vulkan_logits))) in cpu.iter().zip(vulkan).enumerate() {
            assert_eq!(cpu_logits.len(), vulkan_logits.len(), "prompt {prompt:?} step {step} logits shape");
            let vocab_size = cpu_logits.len() / cpu_input.len();
            for (row, (reference, actual)) in
                cpu_logits.chunks(vocab_size).zip(vulkan_logits.chunks(vocab_size)).enumerate()
            {
                let pairs = || reference.iter().zip(actual).map(|(&r, &a)| (f64::from(r), f64::from(a)));
                let error_energy = pairs().map(|(r, a)| (r - a).powi(2)).sum::<f64>();
                let reference_energy = pairs().map(|(r, _)| r * r).sum::<f64>();
                let max_error = pairs().fold(0.0, |max, (r, a)| f64::max(max, (r - a).abs()));
                let max_reference = pairs().fold(0.0, |max, (r, _)| f64::max(max, r.abs()));
                let total_variation =
                    0.5 * softmax(reference).iter().zip(softmax(actual)).map(|(p, q)| (p - q).abs()).sum::<f64>();
                // 0/0 is NaN, which f64::max maps to 0; nonzero/0 stays Inf.
                let ratio = |error: f64, scale: f64| (error / scale).max(0.0);
                let metrics = format!(
                    "prompt={prompt:?} step={step} row={row} rel_l2={:.5} max_normalized={:.5} tv={total_variation:.5}",
                    ratio(error_energy.sqrt(), reference_energy.sqrt()),
                    ratio(max_error, max_reference),
                );
                assert!(
                    max_error <= max_error_limit * max_reference
                        && error_energy <= relative_l2_limit * relative_l2_limit * reference_energy
                        && total_variation <= total_variation_limit,
                    "{metrics} exceeds limits max_normalized={max_error_limit} rel_l2={relative_l2_limit} \
                     tv={total_variation_limit}"
                );
                println!("MODEL_COMPARISON {metrics}");
            }
        }
    }
}

#[uzu_test]
#[ignore]
fn test_full_model_cpu_vulkan_stream_smoke() {
    let model_path = get_test_model_path();
    let vocab_size = u64::from(load_config(&model_path).decoder_config.vocab_size);
    let cpu = run_prompts::<Cpu>(&model_path);
    let vulkan = run_prompts::<Vulkan>(&model_path);
    for (prompt, (cpu, vulkan)) in PROMPTS.iter().zip(cpu.iter().zip(&vulkan)) {
        for (output, tokens) in cpu.iter().chain(vulkan).enumerate() {
            let valid = !tokens.is_empty() && tokens.iter().all(|&token| token < vocab_size);
            assert!(valid, "prompt {prompt:?} output {output} cpu {cpu:?} vulkan {vulkan:?}");
        }
    }
}
