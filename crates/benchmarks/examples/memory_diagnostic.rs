use std::{env, path::PathBuf};

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    parameter::{ContextLength, SamplingMethod, SamplingPolicy},
    types::{Input, Message, Output},
};

#[cfg(target_os = "macos")]
fn get_rss_bytes() -> u64 {
    use std::mem;

    use mach2::{
        kern_return::KERN_SUCCESS,
        mach_types::task_t,
        message::mach_msg_type_number_t,
        task::task_info,
        task_info::{TASK_BASIC_INFO, task_basic_info},
        traps::mach_task_self,
    };

    unsafe {
        let task: task_t = mach_task_self();
        let mut info = task_basic_info {
            virtual_size: 0,
            resident_size: 0,
            user_time: mem::zeroed(),
            system_time: mem::zeroed(),
            policy: 0,
            suspend_count: 0,
        };
        let mut count: mach_msg_type_number_t =
            (mem::size_of::<task_basic_info>() / mem::size_of::<u32>()) as u32;
        let result =
            task_info(task, TASK_BASIC_INFO, &mut info as *mut _ as *mut i32, &mut count);
        if result == KERN_SUCCESS {
            info.resident_size as u64
        } else {
            0
        }
    }
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn print_rss(label: &str) -> u64 {
    let rss = get_rss_bytes();
    println!("[RSS] {label}: {:.1} MB", mb(rss));
    rss
}

fn make_input() -> Input {
    Input::Messages(vec![Message::user(
        "Briefly explain what a neural network is in two sentences.".to_string(),
    )])
}

fn main() {
    let model_path = env::args()
        .nth(1)
        .expect("Usage: memory_diagnostic <model_path>");
    let num_iterations: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    println!("=== Memory Diagnostic ===");
    println!("Model: {model_path}");
    println!("Iterations: {num_iterations}");
    println!();

    let rss_baseline = print_rss("baseline (before anything)");

    for iteration in 0..num_iterations {
        println!("\n--- Iteration {}/{num_iterations} ---", iteration + 1);

        let rss_before_load = print_rss("before model load");

        let decoding_config = DecodingConfig::default().with_context_length(ContextLength::Custom(2048));
        let mut session =
            ChatSession::new(PathBuf::from(&model_path), decoding_config).expect("Failed to create session");

        let rss_after_load = print_rss("after model load");

        // Warmup
        let warmup_config = RunConfig::default().tokens_limit(1);
        session
            .run(make_input(), warmup_config, Some(|_: Output| true))
            .expect("Warmup failed");

        let rss_after_warmup = print_rss("after warmup");

        // Run 5 inference passes
        for pass in 1..=5 {
            let run_config = RunConfig::default()
                .tokens_limit(64)
                .sampling_policy(SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                });
            let output = session
                .run(make_input(), run_config, Some(|_: Output| true))
                .expect("Inference failed");
            let _rss = print_rss(&format!("after inference pass {pass}"));
            println!(
                "  generated {} tokens, output: {:?}",
                output.stats.total_stats.tokens_count_output,
                &output.text.original[..output.text.original.len().min(80)]
            );
        }

        let rss_final = print_rss("before drop");

        drop(session);

        let rss_after_drop = print_rss("after drop");

        println!("\n  Summary (iteration {}):", iteration + 1);
        println!("    Baseline:      {:.1} MB", mb(rss_baseline));
        println!("    Before load:   {:.1} MB (+{:.1} MB from baseline)", mb(rss_before_load), mb(rss_before_load - rss_baseline));
        println!("    After load:    {:.1} MB (+{:.1} MB from before load)", mb(rss_after_load), mb(rss_after_load.saturating_sub(rss_before_load)));
        println!("    After warmup:  {:.1} MB (+{:.1} MB from after load)", mb(rss_after_warmup), mb(rss_after_warmup.saturating_sub(rss_after_load)));
        println!("    Final:         {:.1} MB (+{:.1} MB from after warmup)", mb(rss_final), mb(rss_final.saturating_sub(rss_after_warmup)));
        println!("    After drop:    {:.1} MB ({:.1} MB freed)", mb(rss_after_drop), mb(rss_final.saturating_sub(rss_after_drop)));
    }

    println!("\n=== Done ===");
}
