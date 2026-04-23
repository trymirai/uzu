//! `uzu classify <model> <text>` — run a one-shot classification pass.
//!
//! Designed for both pooled classifiers (BERT-style, single prediction) and
//! per-token classifiers (e.g. openai/privacy-filter, BIOES PII labels).
//!
//! Per-token output prints one line per input token with its top-1 label and
//! softmax confidence — useful for eyeballing numerics against lalamo's
//! expected output.

use std::path::PathBuf;

use console::Style;
use uzu::session::{ClassificationSession, types::Input};

pub fn handle_classify(
    model_path: String,
    message: String,
    top_k: usize,
) {
    let style_bold = Style::new().bold();

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf.file_name().and_then(|s| s.to_str()).unwrap_or("<unknown>");

    println!("Loading {} ...", style_bold.apply_to(model_name));
    let mut session = match ClassificationSession::new(model_path_buf) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load classifier: {:?}", e);
            std::process::exit(1);
        },
    };

    let output = match session.classify(Input::Text(message)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Classification failed: {:?}", e);
            std::process::exit(1);
        },
    };

    let stats = &output.stats;
    println!(
        "\nforward: {:.3}s  post: {:.3}s  total: {:.3}s  tokens: {}  t/s: {:.2}",
        stats.forward_pass_duration,
        stats.postprocessing_duration,
        stats.total_duration,
        stats.tokens_count,
        stats.tokens_per_second,
    );
    println!("rows: {}  labels: {}", output.num_rows, output.num_labels);

    if let Some(per_token) = output.per_token_top1.as_ref() {
        println!("\nper-token top-1:");
        for (idx, (label, conf)) in per_token.iter().enumerate() {
            println!("  [{:>4}] {:<24} {:.4}", idx, label, conf);
        }
    } else {
        // Pooled path: show top-k probabilities.
        let mut probs: Vec<_> = output.probabilities.iter().collect();
        probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        println!("\ntop-{} labels (sigmoid):", top_k);
        for (label, prob) in probs.into_iter().take(top_k) {
            println!("  {:<24} {:.4}", label, prob);
        }
    }
}
