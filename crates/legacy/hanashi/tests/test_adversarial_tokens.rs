mod helpers;

use std::sync::Arc;

use hanashi::{
    Encoding as _,
    chat::hanashi::{Error, HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::{basic::TokenId, session::chat::ChatMessage};
use tokenizers::Tokenizer;

fn sweep(
    config: HanashiConfig,
    tokenizer: Tokenizer,
) {
    let ids: Vec<_> = tokenizer.get_vocab(true).into_values().collect();
    let framing: Vec<_> = config
        .resolve()
        .unwrap()
        .parsing
        .framing_config()
        .tokens
        .iter()
        .map(|text| tokenizer.token_to_id(text).unwrap())
        .collect();
    let eot = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut encoding = HanashiEncodingImpl::new(config, Arc::new(tokenizer)).unwrap();
    let history = vec![ChatMessage::user().with_text("Continue.".into())];
    let reasoning = ["Let me ", "check.", "</think>", "\n\n", "The answer is ", "40."];
    let call = [
        "<tool_call>\n<function=inspect>\n<parameter=options>\n{\"path\":\"",
        "src\"}\n</parameter>\n</function>\n</tool_call>",
    ];
    let combined: Vec<_> = reasoning.iter().chain(&call).copied().chain(["  "]).collect();
    for parts in [&reasoning[..], &call, &combined] {
        let mut positions = vec![0];
        let mut completion = Vec::new();
        for part in parts {
            completion.extend(encoding.tokenize(part).unwrap());
            positions.push(completion.len());
        }
        completion.push(eot);
        encoding.reset().unwrap();
        encoding.encode(history.clone()).unwrap();
        encoding.decode(completion.clone()).unwrap();
        let mut check = |insertions: &[(usize, TokenId)]| {
            let mut tokens = completion.clone();
            for &(position, id) in insertions.iter().rev() {
                tokens.insert(position, id);
            }
            encoding.reset().unwrap();
            encoding.encode(history.clone()).unwrap();
            for id in tokens {
                let result = encoding.decode(vec![id]);
                assert!(!matches!(result, Err(Error::Synchronization(_))), "{insertions:?}, token {id}: {result:?}");
                if result.is_err() || id == eot {
                    break;
                }
            }
        };
        for (index, &position) in positions.iter().enumerate() {
            for &id in &ids {
                check(&[(position, id)]);
            }
            for &second in &positions[index + 1..] {
                for &first_id in framing.iter().filter(|&&id| id != eot) {
                    for &second_id in &framing {
                        check(&[(position, first_id), (second, second_id)]);
                    }
                }
            }
        }
    }
}

#[test]
fn byte_tokens_do_not_desynchronize() {
    let config = HanashiConfig::Qwen35;
    let tokenizer = helpers::byte_tokenizer(&mut config.resolve().unwrap());
    sweep(config, tokenizer);
}

#[test]
#[ignore = "Full vocabulary sweep; set UZU_FUZZ_TOKENIZER and run with --ignored"]
fn qwen35_vocabulary_does_not_desynchronize() {
    let path = std::env::var("UZU_FUZZ_TOKENIZER").unwrap();
    sweep(HanashiConfig::Qwen35, Tokenizer::from_file(path).unwrap());
}
