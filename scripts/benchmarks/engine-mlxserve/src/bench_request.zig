const std = @import("std");
const bench = @import("bench.zig");

pub fn validate(request: bench.BenchRequest) !void {
    if (request.messages.len == 0) return error.EmptyPrompt;
}

pub fn samplingParams(config: bench.BenchSampling) !bench.StochasticSampling {
    return switch (config) {
        .greedy => .{ .temp = 0, .top_k = 1, .top_p = null, .min_p = null, .repetition_penalty = null, .suffix_repetition_length = null },
        .stochastic => |sampling| blk: {
            if (sampling.temp == 0) return error.ZeroTemperatureUseGreedy;
            if ((sampling.min_p orelse 0) != 0) return error.MinPSamplingUnsupported;
            if (sampling.repetition_penalty != null or sampling.suffix_repetition_length != null) return error.RepetitionSamplingUnsupported;
            break :blk sampling;
        },
    };
}

pub fn speculativeDepth(configured: ?usize, default_depth: u32, max_depth: u32) !u32 {
    const depth = configured orelse default_depth;
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
