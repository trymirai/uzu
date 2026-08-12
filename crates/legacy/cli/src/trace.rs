use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use backend_uzu::{ChatCodecConfig, trace::record_trace};
use hanashi::chat::hanashi::renderer::{
    RAISE_EXCEPTION_FUNCTION_NAME, STRFTIME_NOW_FUNCTION_NAME, TEMPLATE_NAME, TOJSON_FILTER_NAME, raise_exception,
    strftime_now, to_json,
};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

struct Prompt {
    template: String,
    rendered: String,
    request: Value,
    token_ids: Vec<u64>,
    tokens: Vec<String>,
}

pub async fn run_trace(
    model_path: String,
    message: String,
    output_path: String,
) -> Result<()> {
    let model_path = Path::new(&model_path);
    let prompt = encode_prompt(model_path, &message)?;
    println!("Prompt tokenized to {} tokens", prompt.token_ids.len());

    let output_path = Path::new(&output_path);
    let output = record_trace(model_path, &prompt.token_ids, output_path, Some(prompt.metadata()?))?;

    println!("Recorded {} arrays to {}", output.array_count, output_path.display());

    Ok(())
}

fn encode_prompt(
    model_path: &Path,
    message: &str,
) -> Result<Prompt> {
    let config: Value = serde_json::from_reader(fs::File::open(model_path.join("config.json"))?)
        .context("Failed to parse config.json")?;
    let codec_config = config.get("token_codec_config").context("config.json has no token_codec_config")?;
    let codec: ChatCodecConfig =
        serde_json::from_value(codec_config.clone()).context("Failed to parse token_codec_config")?;

    let mut messages = Vec::new();
    if let Some(system_prompt) = &codec.default_system_prompt {
        messages.push(json!({ "role": codec.system_role_name, "content": system_prompt }));
    }
    messages.push(json!({ "role": codec.user_role_name, "content": message }));

    let request = json!({
        "add_generation_prompt": true,
        "messages": messages,
        "bos_token": codec.bos_token,
        "eos_token": codec.eos_token,
        "enable_thinking": true,
    });
    let rendered = render(&codec.prompt_template, &request)?;

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("Failed to load tokenizer: {error}"))?;
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|error| anyhow::anyhow!("Failed to tokenize prompt: {error}"))?;
    if encoding.is_empty() {
        bail!("Prompt tokenized to zero tokens");
    }

    Ok(Prompt {
        template: codec.prompt_template,
        rendered,
        request,
        token_ids: encoding.get_ids().iter().map(|token_id| *token_id as u64).collect(),
        tokens: encoding.get_tokens().to_vec(),
    })
}

// Context matches lalamo's ChatCodec.render_request so both sides tokenize the same text.
fn render(
    template: &str,
    request: &Value,
) -> Result<String> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function(STRFTIME_NOW_FUNCTION_NAME, strftime_now);
    environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
    environment.add_filter(TOJSON_FILTER_NAME, to_json);
    environment.add_template(TEMPLATE_NAME, template).context("Invalid prompt template")?;

    let rendered = environment
        .get_template(TEMPLATE_NAME)
        .expect("template was just added")
        .render(context!(..minijinja::Value::from_serialize(request)))
        .context("Failed to render prompt template")?;

    Ok(rendered)
}

impl Prompt {
    fn metadata(&self) -> Result<HashMap<String, String>> {
        Ok(HashMap::from([
            ("add_special_tokens".to_owned(), "false".to_owned()),
            ("prompt_template".to_owned(), self.template.clone()),
            ("rendered_request".to_owned(), self.rendered.clone()),
            ("request".to_owned(), serde_json::to_string(&self.request)?),
            ("tokens".to_owned(), serde_json::to_string(&self.tokens)?),
        ]))
    }
}
