use std::time::Instant;

use uuid::Uuid;
use uzu::types::session::chat::ChatReplyStats;

/// Per-request console logging: one line when the request arrives, one when it
/// ends, and optional cache-reset metadata, correlated by a short tag derived
/// from the request id. Each line is a single `println!`, which holds the stdout
/// lock for the whole write, so lines from concurrent requests never interleave.
pub struct RequestLog {
    tag: String,
    started: Instant,
}

fn short_tag(id: &str) -> String {
    let id = id.strip_prefix("chatcmpl-").unwrap_or(id);
    id.chars().take(8).collect()
}

impl RequestLog {
    pub fn start(
        id: &str,
        stream: bool,
        messages: usize,
        tools: usize,
        reasoning_effort: Option<&str>,
    ) -> Self {
        let tag = short_tag(id);
        println!(
            "[req {tag}] received: {messages} messages, {}, {tools} tools, reasoning_effort={}",
            if stream {
                "stream"
            } else {
                "blocking"
            },
            reasoning_effort.unwrap_or("default"),
        );
        Self {
            tag,
            started: Instant::now(),
        }
    }

    /// For requests rejected before an id could be assigned to them.
    pub fn rejected(error: &str) {
        println!("[req {}] rejected: {error}", short_tag(&Uuid::new_v4().simple().to_string()));
    }

    pub fn prefix_cache_reset(
        &self,
        reason: &'static str,
        stored_messages: usize,
        incoming_messages: usize,
    ) {
        println!(
            "[req {}] prefix cache reset: reason={reason}, stored_messages={stored_messages}, incoming_messages={incoming_messages}",
            self.tag,
        );
    }

    pub fn finish(
        &self,
        finish_reason: &str,
        stats: Option<&ChatReplyStats>,
        notes: Vec<String>,
    ) {
        let mut parts =
            vec![format!("completed in {:.2}s, finish={finish_reason}", self.started.elapsed().as_secs_f64())];
        if let Some(stats) = stats {
            let cached = stats.tokens_count_input_cached.unwrap_or(0);
            let prefilled = stats.tokens_count_input.unwrap_or(0);
            parts.push(format!(
                "prompt {} tok ({} cached, {} prefilled @ {} tok/s)",
                cached + prefilled,
                cached,
                prefilled,
                stats.prefill_tokens_per_second.map_or("-".to_string(), |tps| format!("{tps:.1}")),
            ));
            if let Some(ttft) = stats.time_to_first_token {
                parts.push(format!("ttft {ttft:.2}s"));
            }
            if let Some(generated) = stats.tokens_count_output {
                parts.push(format!(
                    "decode {generated} tok @ {} tok/s",
                    stats.generate_tokens_per_second.map_or("-".to_string(), |tps| format!("{tps:.1}")),
                ));
            }
            if let Some(speculator) = &stats.speculator_stats {
                parts.push(format!(
                    "spec {:.2} tok/pass ({} passes)",
                    speculator.tokens_per_forward_pass, speculator.num_decode_forward_passes
                ));
            }
        }
        parts.extend(notes);
        println!("[req {}] {}", self.tag, parts.join(", "));
    }

    pub fn fail(
        &self,
        error: &str,
    ) {
        println!("[req {}] failed in {:.2}s: {error}", self.tag, self.started.elapsed().as_secs_f64());
    }
}
