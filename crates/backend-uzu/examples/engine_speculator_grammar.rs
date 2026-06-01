use std::{fs::File, path::PathBuf, time::Instant};

use backend_uzu::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{
            grammar::{Grammar, GrammarConfig},
            stream::{LanguageModelStreamSpeculatorOptions, TrieCreationConfig},
        },
    },
    speculators::prompt_lookup_speculator::PromptLookupSpeculator,
};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct City {
    name: String,
    description: String,
}

pub fn main() {
    let model_path = PathBuf::from(std::env::args().nth(1).unwrap());
    let prompt = "Return a JSON object for a city of London (name, description). Description must be a comperhensive overview, ~250 words";

    let config: Value = serde_json::from_reader(File::open(model_path.join("config.json")).unwrap()).unwrap();
    let codec = &config["token_codec_config"];
    let stop_token_ids = config["generation_config"]["stop_token_ids"]
        .as_array()
        .unwrap()
        .into_iter()
        .map(|v| v.as_u64().unwrap())
        .collect::<Vec<u64>>();
    let stop_token_ids_i32 = stop_token_ids.iter().map(|&token| token as i32).collect::<Vec<_>>();
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();

    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_template("chat_template", codec["prompt_template"].as_str().unwrap()).unwrap();
    let prompt = environment
        .get_template("chat_template")
        .unwrap()
        .render(context!(
            messages => [json!({
                "role": codec["user_role_name"].as_str().unwrap_or("user"),
                "content": prompt,
            })],
            add_generation_prompt => true,
            bos_token => codec["bos_token"].as_str(),
            eos_token => codec["eos_token"].as_str(),
            enable_thinking => false,
        ))
        .unwrap();

    let prompt = tokenizer
        .encode(prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&token| u64::from(token))
        .collect::<Vec<_>>();

    let grammar_config = GrammarConfig::from_json_schema_type::<City>().unwrap();
    let grammar = <dyn Grammar>::new(&grammar_config, &tokenizer, None, Some(&stop_token_ids_i32)).unwrap();

    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_language_model(&model_path).unwrap();
    let mut state = model.create_empty_state(model.recommended_context_length()).unwrap();
    let speculator = PromptLookupSpeculator::new();
    let mut options = model.default_stream_options();
    options.grammar = Some(grammar);
    options.speculator = Some(LanguageModelStreamSpeculatorOptions {
        speculator: &speculator,
        speculation_budget: 16,
        trie_creation_config: TrieCreationConfig::default(),
    });
    let end_to_end_start = Instant::now();
    let mut stream = model.stream(&prompt, &mut state, options).unwrap().take(2048);

    let mut output = Vec::new();
    let token = stream.next().unwrap();
    let Ok(token) = token else {
        eprintln!("ERROR {:?}", token.unwrap_err());
        return;
    };
    output.push(token as u32);

    let decode_start = Instant::now();
    if !stop_token_ids.contains(&token) {
        for token in stream {
            let Ok(token) = token else {
                eprintln!("ERROR {:?}", token.unwrap_err());
                break;
            };
            output.push(token as u32);
            if stop_token_ids.contains(&token) {
                break;
            }
        }
    }
    let decode_elapsed = decode_start.elapsed();
    let end_to_end_elapsed = end_to_end_start.elapsed();
    let decode_only_tps = (output.len() - 1) as f64 / decode_elapsed.as_secs_f64();
    let end_to_end_tps = output.len() as f64 / end_to_end_elapsed.as_secs_f64();

    println!("{}", tokenizer.decode(&output, false).unwrap());
    println!("{} decode-only tps", decode_only_tps);
    println!("{} end-to-end tps", end_to_end_tps);
}
