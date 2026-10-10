const std = @import("std");

// Keep these wire models in parity with Rust, C++, and Python (AGENTS.md).
pub const ChatMessage = struct {
    role: []const u8,
    content: ?[]const u8 = null,
    reasoning_content: ?[]const u8 = null,
    tool_calls: ?[]const std.json.Value = null,
    tool_call_id: ?[]const u8 = null,
};

pub const BenchSampling = struct {
    top_k: ?i32 = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    temp: ?f32 = null,
};

pub const BenchRequest = struct {
    prompt_text: ?[]const u8 = null,
    prompt_chat: ?[]const ChatMessage = null,
    tools: ?[]const std.json.Value = null,
    tool_choice: ?std.json.Value = null,
    max_tokens: ?usize = null,
    speculative_depth: ?usize = null,
    sampling: ?BenchSampling = null,
    num_runs: ?usize = null,
};

pub const BenchResponse = struct {
    text: []const u8,
    tokens_count: usize,
    time_to_first_token: f64,
    prompt_tps: f64,
    decode_tps: f64,
    tokens_per_forward_pass: f64,
    duration: f64,
    memory_phys_footprint: u64,
    memory_resident: u64,
    memory_graphics_total: u64,
};
