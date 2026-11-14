const std = @import("std");
const c = @import("system_c");
pub const mlx = @import("mlx/ffi.zig");
pub const log = @import("sdk/util/log.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const env = @import("env.zig");
const model = @import("sdk/config/model.zig");
const tokenizer = @import("sdk/tokenizer/tokenizer.zig");
const chat = @import("host/chat/chat.zig");
const transformer = @import("engines/mlx/transformer.zig");
const generate = @import("engines/mlx/generate.zig");
const bench = @import("bench.zig");
const request_utils = @import("bench_request.zig");
const decode = @import("bench_decode.zig");
const MemoryPeak = @import("bench_memory.zig").MemoryPeak;

const Engine = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    config: model.ModelConfig,
    tok: tokenizer.Tokenizer,
    chat_config: chat.ChatConfig,
    weights: model.Weights,
    xfm: transformer.Transformer,
    speculation: decode.Speculation,
    sampling: generate.SamplingParams,

    fn init(io: std.Io, allocator: std.mem.Allocator, path: []const u8, draft_path: ?[]const u8, speculative_depth: ?usize, sampling: bench.StochasticSampling) !*Engine {
        // Retain stable addresses for weights and all model-lifetime state.
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        self.sampling = .{
            .temperature = sampling.temp orelse 1,
            .top_p = sampling.top_p orelse 1,
            .top_k = @intCast(sampling.top_k orelse 0),
        };
        self.io = io;
        self.allocator = allocator;
        self.config = try model.parseConfig(io, allocator, path);
        errdefer self.config.deinit(allocator);

        self.tok = try tokenizer.loadTokenizer(io, allocator, path);
        errdefer self.tok.deinit();

        self.chat_config = try chat.loadChatConfig(io, allocator, path);
        errdefer self.chat_config.deinit();

        // Mirrors main.zig's offline MLX loader at the pinned revision.
        // Benchmark inputs must not depend on the user's app settings file.
        self.config.applyTokenizer(&self.tok, self.chat_config.eos_token);
        if (self.config.use_bidirectional_attention) return error.EmbeddingModelUnsupported;
        const served = self.config.plugin != null;
        self.weights = if (served) model.Weights.init(allocator) else try model.loadModelWeights(io, allocator, path, &self.config, false);
        errdefer self.weights.deinit();

        if (!served) {
            model.resolveWeightPrefix(&self.config, &self.weights);
            try model.narrowHadamardPackTables(&self.config, &self.weights, mlx.gpuStream());
        }
        self.xfm = if (served)
            try @import("engines/mlx/plugin/driver.zig").load(io, allocator, &self.config, path, mlx.gpuStream())
        else
            try transformer.Transformer.init(io, allocator, self.config);
        errdefer self.xfm.deinit();

        self.xfm.round_cost.layout = @import("engines/mlx/spec/round_cost.zig").layoutFor(&self.config);
        generate.installSuppressMask(&self.xfm, &self.tok, self.chat_config.chat_template, self.config.eosTokenSlice());
        _ = mlx.applyWiredPolicy();
        if (self.config.hidden_act == .gelu_approx) {
            self.xfm.compileGelu();
            self.xfm.compileGeglu();
        }
        if (self.xfm.moe_layers != null) {
            self.xfm.compileMoeRouting();
        }
        if (self.config.linear_num_key_heads > 0) {
            self.xfm.compileGdnGate();
        }

        try self.speculation.init(io, allocator, &self.xfm, path, draft_path, speculative_depth);
        errdefer self.speculation.deinit();

        // Loading may schedule lazy GPU work. Keep it out of the first request.
        try mlx.check(mlx.mlx_synchronize(self.xfm.s));
        return self;
    }

    fn deinit(self: *Engine) void {
        self.speculation.deinit();
        self.xfm.deinit();
        self.weights.deinit();
        self.chat_config.deinit();
        self.tok.deinit();
        self.config.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn tokenize(self: *Engine, a: std.mem.Allocator, request: bench.BenchRequest) ![]u32 {
        const input = request.messages;
        const messages = try a.alloc(chat.Message, input.len);

        for (input, messages) |message, *out| {
            out.* = .{
                .role = message.role,
                .content = message.content orelse "",
                .reasoning_content = message.reasoning_content,
                .tool_call_id = message.tool_call_id,
            };
            if (message.tool_calls) |calls| {
                const converted = try a.alloc(chat.ToolCall, calls.len);
                for (calls, converted) |call, *dest| {
                    const function = try objectField(call, "function");
                    const arguments = try objectField(function, "arguments");
                    dest.* = .{
                        .id = try stringValue(try objectField(call, "id")),
                        .name = try stringValue(try objectField(function, "name")),
                        .arguments = if (arguments == .string) arguments.string else try std.json.Stringify.valueAlloc(a, arguments, .{}),
                    };
                }
                out.tool_calls = converted;
            }
        }

        var tools_json: ?[]const u8 = if (request.tools) |tools| try std.json.Stringify.valueAlloc(a, tools, .{}) else null;
        var instruction: ?[]const u8 = null;

        if (request.tool_choice) |choice| {
            if (choice == .string) {
                if (std.mem.eql(u8, choice.string, "none")) {
                    tools_json = null;
                } else if (std.mem.eql(u8, choice.string, "required")) {
                    instruction = "\nYou MUST call one of the available functions. Do not respond with text.";
                } else if (!std.mem.eql(u8, choice.string, "auto")) {
                    return error.InvalidToolChoice;
                }
            } else if (choice == .object) {
                const name = try stringValue(try objectField(try objectField(choice, "function"), "name"));
                instruction = try std.fmt.allocPrint(a, "\nYou MUST call the function \"{s}\". Do not respond with text.", .{name});
            } else {
                return error.InvalidToolChoice;
            }
        }
        const thinking = try self.checkedThinking(a, messages, tools_json, instruction, request.reasoning_effort);
        return chat.formatChat(a, &self.tok, messages, &self.chat_config, tools_json, instruction, thinking.enable, thinking.effort, false);
    }

    const Thinking = struct { enable: bool, effort: ?[]const u8 };

    fn nativeThinking(self: *Engine, effort: bench.ReasoningEffort, tools: bool) Thinking {
        return switch (effort) {
            .default => .{ .enable = self.config.defaultEnableThinking(tools), .effort = null },
            .disabled => .{ .enable = false, .effort = "none" },
            else => .{ .enable = true, .effort = @tagName(effort) },
        };
    }

    fn checkedThinking(self: *Engine, a: std.mem.Allocator, messages: []const chat.Message, tools_json: ?[]const u8, instruction: ?[]const u8, effort: bench.ReasoningEffort) !Thinking {
        const requested = self.nativeThinking(effort, tools_json != null);
        if (effort == .default) return requested;

        const prompt = try chat.renderChatTemplate(a, messages, &self.chat_config, tools_json, instruction, requested.enable, requested.effort, false);
        var honored = false;
        var unique = true;
        for (std.enums.values(bench.ReasoningEffort)) |other| {
            if (other == effort) continue;
            const alternative = self.nativeThinking(other, tools_json != null);
            const rendered = try chat.renderChatTemplate(a, messages, &self.chat_config, tools_json, instruction, alternative.enable, alternative.effort, false);
            const same = std.mem.eql(u8, prompt, rendered);
            if (!same) honored = true;
            if (same and other != .disabled and other != .default) unique = false;
        }
        if (if (effort == .disabled) !honored else !unique) return error.ReasoningEffortUnsupported;
        return requested;
    }

    fn execute(self: *Engine, a: std.mem.Allocator, request: bench.BenchRequest) !bench.BenchResponse {
        try request_utils.validate(request);
        const prompt = try self.tokenize(a, request);
        const limit = try request_utils.tokenLimit(request.max_output_tokens, prompt.len, self.config.contextCap());
        const options = self.speculation.options(&self.xfm);
        var sampling = self.sampling;
        if (request.seed) |seed| sampling.seed = seed;
        return self.runOne(a, prompt, limit, sampling, options);
    }

    fn runOne(self: *Engine, a: std.mem.Allocator, prompt: []const u32, limit: u32, sampling: generate.SamplingParams, options: generate.Generator.InitOptions) !bench.BenchResponse {
        try mlx.check(mlx.mlx_synchronize(self.xfm.s));
        try self.xfm.resetCache();
        self.speculation.reset(&self.xfm);
        errdefer _ = mlx.mlx_synchronize(self.xfm.s);
        try mlx.check(mlx.mlx_reset_peak_memory());

        var memory = try MemoryPeak.init();
        memory.start();
        defer memory.stop();

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(a);

        const energy_start = try gpuEnergy();
        const timer = io_util.Stopwatch.init(self.io);
        var gen = try generate.Generator.initWithOptions(self.io, a, &self.xfm, &self.tok, prompt, limit, sampling, self.config.eosTokenSlice(), options);
        // Even an error must drain GPU work before the next request resets it.
        defer {
            _ = mlx.mlx_synchronize(self.xfm.s);
            gen.deinit(a);
        }

        var forwards: usize = 0;
        self.xfm.benchmark_target_forwards = &forwards;
        defer self.xfm.benchmark_target_forwards = null;

        try memory.sample();
        var first_ns: ?u64 = null;
        var last_ns: ?u64 = null;
        var first_batch_tokens: usize = 0;
        var single: [1]u32 = undefined;
        generation: while (try decode.next(&gen, a, &single)) |batch| {
            defer if (batch.owned) a.free(batch.tokens);
            const now = timer.read();
            const first_batch = first_ns == null;
            try mlx.checkErrorDecode();
            try memory.sample();
            for (batch.tokens) |id| {
                // A speculative batch can contain EOS before its last token.
                if (generate.isEosId(id, self.config.eosTokenSlice())) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    break :generation;
                }
                if (first_ns == null) {
                    first_ns = now;
                }

                last_ns = now;
                try ids.append(a, id);
                if (first_batch) first_batch_tokens += 1;
                if (ids.items.len == limit) {
                    gen.done = true;
                    gen.finish_reason = "length";
                    break :generation;
                }
            }
        }
        try mlx.check(mlx.mlx_synchronize(self.xfm.s));
        try mlx.checkError();
        const finished_ns = timer.read();
        const energy_gpu = try gpuEnergy() - energy_start;
        try memory.sample();
        // DSpark verifies directly in deepseek_v4.zig, bypassing forwardWith.
        // Each completed round has one target verify; rollback uses saved state.
        forwards += @intCast(gen.dspark_attempted);
        gen.logSpecStats();

        var memory_used: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&memory_used));
        const ttft = request_utils.seconds(first_ns orelse finished_ns);
        const decode_seconds = if (first_ns) |first| request_utils.seconds(last_ns.? - first) else 0;
        return .{
            .text = try self.tok.decode(a, ids.items, self.tok.tok_type == .sentencepiece_bpe),
            .finish_reason = std.meta.stringToEnum(bench.FinishReason, gen.finish_reason) orelse return error.UnexpectedFinishReason,
            .input_tokens_count = prompt.len,
            .output_tokens_count = ids.items.len,
            .time_to_first_token = ttft,
            .prefill_tps = request_utils.rate(prompt.len, ttft),
            .decode_tps = request_utils.rate(ids.items.len - first_batch_tokens, decode_seconds),
            .tokens_per_forward_pass = if (decode.speculative(&gen) and forwards > 0)
                request_utils.rate(ids.items.len -| 1, @floatFromInt(forwards))
            else
                null,
            .duration = request_utils.seconds(finished_ns),
            .energy_gpu = energy_gpu,
            .memory_used = memory_used,
            .memory_phys_footprint = memory.snapshot.phys_footprint,
            .memory_resident = memory.snapshot.resident_size,
            .memory_graphics_total = memory.snapshot.graphics_total,
        };
    }
};

fn gpuEnergy() !f64 {
    var joules: f64 = undefined;
    if (c.get_gpu_energy(&joules) != c.KERN_SUCCESS) return error.GpuEnergyUnavailable;
    return joules;
}

fn objectField(value: std.json.Value, key: []const u8) !std.json.Value {
    if (value != .object) {
        return error.InvalidToolCall;
    }
    return value.object.get(key) orelse error.InvalidToolCall;
}

fn stringValue(value: std.json.Value) ![]const u8 {
    if (value != .string) {
        return error.InvalidToolCall;
    }
    return value.string;
}

pub fn main(init: std.process.Init) !void {
    env.init();
    const a = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer args.deinit();

    _ = args.next();
    var config_argument: ?[]const u8 = null;
    var model_argument: ?[]const u8 = null;
    var draft_argument: ?[]const u8 = null;
    while (args.next()) |flag| {
        const value = args.next() orelse return error.ExpectedValue;
        if (std.mem.eql(u8, flag, "--config")) {
            config_argument = value;
        } else if (std.mem.eql(u8, flag, "--model")) {
            model_argument = value;
        } else if (std.mem.eql(u8, flag, "--draft-model")) {
            draft_argument = value;
        } else {
            return error.UnexpectedArgument;
        }
    }

    const ModelConfig = struct {
        model: []const u8,
        specdec: bool,
        revision: ?[]const u8 = null,
        draft_model: ?[]const u8 = null,
        draft_revision: ?[]const u8 = null,
        speculative_depth: ?usize = null,
        sampling: bench.BenchSampling,
        reasoning_efforts: bench.ReasoningEfforts,
    };
    const config_json = try std.Io.Dir.cwd().readFileAlloc(io, config_argument orelse return error.ExpectedConfig, a, .limited(1024 * 1024));
    defer a.free(config_json);
    const config = try std.json.parseFromSlice(ModelConfig, a, config_json, .{});
    defer config.deinit();
    const sampling = try request_utils.samplingParams(config.value.sampling);

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, model_argument orelse return error.ExpectedModel, a);
    defer a.free(path);

    const draft_path = if (draft_argument) |p| try std.Io.Dir.cwd().realPathFileAlloc(io, p, a) else null;
    defer if (draft_path) |p| a.free(p);

    const output_fd = c.dup(1);
    if (output_fd < 0) {
        return error.StdoutRedirectFailed;
    }
    const output: std.Io.File = .{ .handle = output_fd, .flags = std.Io.File.stdout().flags };
    defer output.close(io);

    if (c.dup2(2, 1) < 0) {
        return error.StdoutRedirectFailed;
    }

    mlx.installErrorHandler();
    @import("mlx/state.zig").applyMlxCacheLimit();
    @import("host/registry/memory_ceiling.zig").applyGpuCeilingEnv();
    const device = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(device);

    try mlx.check(mlx.mlx_set_default_device(device));
    const engine = try Engine.init(io, a, path, draft_path, config.value.speculative_depth, sampling);
    defer engine.deinit();
    if (config.value.specdec != (engine.speculation.depth > 0)) {
        return error.SpecdecMismatch;
    }

    log.info("mlx-serve benchmark ready (v{s}, native MLX, mode={s}, depth={d}, max_depth={d})\n", .{
        @import("build_options").version, @tagName(engine.speculation.mode), engine.speculation.depth, engine.speculation.max_depth,
    });

    // Allow long prompts while bounding each JSONL record at 64 MiB.
    // Stdin reading and JSON work happen outside runOne's timer.
    const input_buffer = try a.alloc(u8, 64 * 1024 * 1024);
    defer a.free(input_buffer);

    var input = std.Io.File.stdin().reader(io, input_buffer);
    var output_buffer: [16 * 1024]u8 = undefined;
    var writer = output.writer(io, &output_buffer);
    try writeLine(&writer.interface, bench.ReadyResponse{});
    while (try input.interface.takeDelimiter('\n')) |line| {
        if (std.mem.trim(u8, line, " \t\r").len == 0) {
            continue;
        }
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();

        const request_type = try std.json.parseFromSlice(bench.RequestType, arena.allocator(), line, .{ .ignore_unknown_fields = true });
        switch (request_type.value.type) {
            .CooldownRequest => {
                const cooldown = try std.json.parseFromSlice(bench.CooldownRequest, arena.allocator(), line, .{});
                const cooldown_result = c.await_cooldown(cooldown.value.temperature, cooldown.value.only_if_capped);
                if (cooldown_result != c.KERN_SUCCESS) {
                    log.err("Cooldown failed ({d})\n", .{cooldown_result});
                    return error.CooldownFailed;
                }
                try writeLine(&writer.interface, bench.CooldownResponse{});
                continue;
            },
            .BenchRequest => {},
        }

        const request = try std.json.parseFromSlice(bench.BenchRequest, arena.allocator(), line, .{});
        try writeLine(&writer.interface, try engine.execute(arena.allocator(), request.value));
    }
}

fn writeLine(writer: *std.Io.Writer, value: anytype) !void {
    try std.json.Stringify.value(value, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}
