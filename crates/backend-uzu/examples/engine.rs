use std::{fs::File, path::PathBuf, time::Instant};

use backend_uzu::{backends::metal::Metal, engine::Engine};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

pub fn main() {
    let model_path = PathBuf::from(std::env::args().nth(1).unwrap());
    let prompt = "Tell me about London";

    let config: Value = serde_json::from_reader(File::open(model_path.join("config.json")).unwrap()).unwrap();
    let codec = &config["token_codec_config"];
    let stop_token_ids = config["generation_config"]["stop_token_ids"]
        .as_array()
        .unwrap()
        .into_iter()
        .map(|v| v.as_u64().unwrap())
        .collect::<Vec<u64>>();
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

    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_language_model(&model_path).unwrap();
    let mut state = model.create_empty_state(model.recommended_context_length()).unwrap();
    let stream = model.stream(&prompt, &mut state, model.default_stream_options()).unwrap();

    let mut output = Vec::new();
    let start = Instant::now();
    for token in stream.take(2048) {
        let token = token.unwrap();
        output.push(token as u32);
        if stop_token_ids.contains(&token) {
            break;
        }
    }
    let elapsed = start.elapsed();

    println!("{}", tokenizer.decode(&output, false).unwrap());
    println!("{} tps", output.len() as f64 / elapsed.as_secs_f64());
}
