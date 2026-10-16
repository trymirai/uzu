const std = @import("std");
const BenchRequest = @import("bench.zig").BenchRequest;
const BenchSampling = @import("bench.zig").BenchSampling;

pub fn validate(request: BenchRequest) !void {
    if ((request.num_runs orelse 1) == 0) {
        return error.InvalidNumRuns;
    }
    if (request.prompt_text) |text| {
        if (text.len == 0) return error.EmptyPrompt;
    } else if (request.prompt_chat) |messages| {
        if (messages.len == 0) return error.EmptyPrompt;
    } else {
        return error.MissingPrompt;
    }

    const s = request.sampling orelse BenchSampling{};
    if ((s.min_p orelse 0) != 0) {
        return error.MinPSamplingUnsupported;
    }
    if (s.top_k) |k| {
        if (k < 0) {
            return error.InvalidTopK;
        }
    }
    if (s.top_p) |p| {
        if (!std.math.isFinite(p) or p < 0 or p > 1) {
            return error.InvalidTopP;
        }
    }
    if (s.temp) |t| {
        if (!std.math.isFinite(t) or t < 0 or t > 2) {
            return error.InvalidTemperature;
        }
    }
}

pub fn speculativeDepth(requested: ?usize, default_depth: u32, max_depth: u32) !u32 {
    const depth = requested orelse default_depth;
    if (depth > max_depth) {
        return error.InvalidSpeculativeDepth;
    }
    return @intCast(depth);
}

pub fn tokenLimit(requested: ?usize, prompt_size: usize, context_size: u32) !u32 {
    if (prompt_size == 0) {
        return error.EmptyPrompt;
    }
    if (prompt_size >= std.math.maxInt(u32)) {
        return error.PromptTooLong;
    }
    if (context_size != 0 and prompt_size >= context_size) {
        return error.ContextExceeded;
    }

    const remaining: usize = if (context_size == 0) std.math.maxInt(u32) else context_size - prompt_size;
    const limit = if ((requested orelse 0) == 0) remaining else requested.?;
    if (limit > remaining or limit > std.math.maxInt(u32)) {
        return error.ContextExceeded;
    }
    return @intCast(limit);
}

pub fn seconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

pub fn rate(count: usize, duration: f64) f64 {
    return if (duration > 0) @as(f64, @floatFromInt(count)) / duration else 0;
}

test "speculative depth defaults, opt-out, and bounds" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 3), try speculativeDepth(null, 3, 8));
    try t.expectEqual(@as(u32, 0), try speculativeDepth(0, 3, 8));
    try t.expectEqual(@as(u32, 8), try speculativeDepth(8, 3, 8));
    try t.expectError(error.InvalidSpeculativeDepth, speculativeDepth(9, 3, 8));
    try t.expectError(error.InvalidSpeculativeDepth, speculativeDepth(std.math.maxInt(usize), 3, 8));
    try t.expectEqual(@as(u32, 0), try speculativeDepth(null, 0, 0));
    try t.expectError(error.InvalidSpeculativeDepth, speculativeDepth(1, 0, 0));
}
