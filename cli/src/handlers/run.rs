use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use inquire::Text;
use uzu::session::{
    sampling_config::SamplingConfig, session_config::SessionRunConfig,
    session_input::SessionInput, session_output::SessionOutput,
};

use crate::server::load_session;

fn format_output(output: SessionOutput) -> String {
    let stats = &output.stats;
    let tokens_per_second = if let Some(generate_stats) = &stats.generate_stats
    {
        generate_stats.tokens_per_second
    } else {
        stats.prefill_stats.tokens_per_second
    };

    let style_stats = Style::new().bold();
    let stats_info = style_stats.apply_to(format!(
        "{:.3}s, {:.3}t/s",
        stats.total_stats.duration, tokens_per_second,
    ));

    let result = format!("{}\n\n{}", output.text, stats_info,);
    result
}

pub fn handle_run(
    model_path: String,
    tokens_limit: usize,
) {
    let mut session = load_session(model_path);

    let is_model_running = Arc::new(AtomicBool::new(false));
    let is_model_running_for_ctrlc = is_model_running.clone();
    ctrlc::set_handler(move || {
        if is_model_running_for_ctrlc.load(Ordering::SeqCst) {
            is_model_running_for_ctrlc.store(false, Ordering::SeqCst);
        }
    })
    .unwrap();

    loop {
        let input =
            match Text::new("").with_placeholder("Send a message").prompt() {
                Ok(input) => input,
                Err(_) => {
                    break;
                },
            };
        if input.is_empty() {
            continue;
        }

        is_model_running.store(true, Ordering::SeqCst);

        let progress_bar_message_limit: usize = 1024;
        let progress_bar = ProgressBar::new_spinner();
        progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));
        progress_bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );

        let progress_bar_for_progress = progress_bar.clone();
        let is_model_running_for_progress = is_model_running.clone();
        let session_progress = move |output: SessionOutput| {
            if !is_model_running_for_progress.load(Ordering::SeqCst) {
                return false;
            }

            let result: String = format_output(output)
                .chars()
                .rev()
                .take(progress_bar_message_limit)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            progress_bar_for_progress.set_message(result);
            return true;
        };

        let session_output = session.run(
            SessionInput::Text(input.to_string()),
            SessionRunConfig::new_with_sampling(
                tokens_limit as u64,
                SamplingConfig::TopP {
                    top_p: 0.95,
                },
            ),
            Some(session_progress),
        );
        let result = format_output(session_output);

        progress_bar.set_style(
            ProgressStyle::default_spinner().template("{msg}").unwrap(),
        );
        progress_bar.finish_and_clear();
        println!("{}", result);
        is_model_running.store(false, Ordering::SeqCst);
    }
}
