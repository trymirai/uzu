use console::Style;
use inquire::Text;
use uzu::session::types::Input;

use crate::server::load_classification_session;

pub fn handle_classify(model_path: String, mut message: Option<String>) {
    let mut session = load_classification_session(model_path);

    let non_interactive = message.is_some();
    let style_stats = Style::new().bold();

    loop {
        let input = if let Some(msg) = message.take() {
            msg
        } else {
            match Text::new("").with_placeholder("Enter text to classify").prompt() {
                Ok(input) => input,
                Err(_) => break,
            }
        };
        if input.is_empty() {
            if non_interactive {
                break;
            }
            continue;
        }

        let output = match session.classify(Input::Text(input)) {
            Ok(output) => output,
            Err(e) => {
                eprintln!("Error during classification: {}", e);
                if non_interactive {
                    return;
                }
                continue;
            },
        };

        for (label, probability) in &output.probabilities {
            println!("{}: {:.4}", label, probability);
        }
        let stats_info = style_stats.apply_to(format!(
            "{:.3}s, {} tokens",
            output.stats.total_duration, output.stats.tokens_count,
        ));
        println!("{}\n", stats_info);

        if non_interactive {
            break;
        }
    }
}
