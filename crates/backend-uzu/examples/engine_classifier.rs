use std::{fs::File, path::PathBuf};

use backend_uzu::{backends::metal::Metal, engine::Engine};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

pub fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().unwrap());
    let input = args.collect::<Vec<_>>().join(" ");
    let input = if input.is_empty() {
        "Hi"
    } else {
        input.as_str()
    };

    let config: Value = serde_json::from_reader(File::open(model_path.join("config.json")).unwrap()).unwrap();
    let codec = &config["token_codec_config"];
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
                "content": input,
            })],
            add_generation_prompt => false,
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
    let model = engine.load_classifier_model(&model_path).unwrap();

    let mut probabilities = model.classify(&prompt).unwrap().probabilities.into_iter().collect::<Vec<_>>();
    probabilities.sort_by(|(left, _), (right, _)| left.cmp(right));

    for (label, probability) in probabilities {
        println!("{label}: {probability:.6}");
    }
}
