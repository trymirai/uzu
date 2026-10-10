const std = @import("std");
pub const mlx = @import("mlx.zig");
pub const log = @import("log.zig");
pub const io_util = @import("io_util.zig");
const model = @import("model.zig");
const tokenizer = @import("tokenizer.zig");
const chat = @import("chat.zig");
const transformer = @import("transformer.zig");
const generate = @import("generate.zig");
const bench = @import("bench.zig");
const request_utils = @import("bench_request.zig");
const decode = @import("bench_decode.zig");
const MemoryPeak = @import("bench_memory.zig").MemoryPeak;

extern "c" fn dup(fd: c_int) c_int;
extern "c" fn dup2(old: c_int, new: c_int) c_int;

const Engine = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    config: model.ModelConfig,
    tok: tokenizer.Tokenizer,
    chat_config: chat.ChatConfig,
    weights: model.Weights,
    xfm: transformer.Transformer,
    speculation: decode.Speculation,

    fn init(io: std.Io, allocator: std.mem.Allocator, path: []const u8, draft_path: ?[]const u8) !*Engine {
        // Retain stable addresses for weights and all model-lifetime state.
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

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
        self.weights = try model.loadModelWeights(io, allocator, path, &self.config, false);
        errdefer self.weights.deinit();

        model.resolveWeightPrefix(&self.config, &self.weights);
        try model.narrowHadamardPackTables(&self.config, &self.weights, mlx.gpuStream());
        self.xfm = try transformer.Transformer.init(io, allocator, self.config, &self.weights);
        errdefer self.xfm.deinit();

        self.xfm.round_cost.layout = @import("round_cost.zig").layoutFor(&self.config);
        generate.installSuppressMask(&self.xfm, &self.tok, self.chat_config.chat_template, self.config.eosTokenSlice());
        _ = mlx.applyWiredPolicy();
        if (self.config.hidden_act == .gelu_approx) {
            self.xfm.compileGelu();
            self.xfm.compileGeglu();
        }
        if (self.config.final_logit_softcapping > 0) {
            self.xfm.compileSoftcap();
        }
        if (self.xfm.moe_layers != null) {
            self.xfm.compileMoeRouting();
        }
        if (self.config.linear_num_key_heads > 0) {
            self.xfm.compileGdnGate();
        }

        try self.speculation.init(io, allocator, &self.xfm, path, draft_path);
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
        if (request.prompt_text) |text| {
            return self.tok.encode(a, text);
        }
        const input = request.prompt_chat orelse {
            return error.MissingPrompt;
        };
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
        return chat.formatChat(a, &self.tok, messages, &self.chat_config, tools_json, instruction, self.config.defaultEnableThinking(tools_json != null), null, false);
    }

    fn execute(self: *Engine, a: std.mem.Allocator, request: bench.BenchRequest) ![]bench.BenchResponse {
        try request_utils.validate(request);
        const prompt = try self.tokenize(a, request);
        const limit = try request_utils.tokenLimit(request.max_tokens, prompt.len, self.config.contextCap());
        const sampling = request.sampling orelse bench.BenchSampling{};
        const params = generate.SamplingParams{
            .temperature = sampling.temp orelse 0,
            .top_p = sampling.top_p orelse 1,
            .top_k = @intCast(sampling.top_k orelse 0),
        };
        const options = try self.speculation.options(&self.xfm, request.speculative_depth);
        const responses = try a.alloc(bench.BenchResponse, request.num_runs orelse 1);
        for (responses) |*response| response.* = try self.runOne(a, prompt, limit, params, options);
        return responses;
    }

    fn runOne(self: *Engine, a: std.mem.Allocator, prompt: []const u32, limit: u32, sampling: generate.SamplingParams, options: generate.Generator.InitOptions) !bench.BenchResponse {
        try mlx.check(mlx.mlx_synchronize(self.xfm.s));
        try self.xfm.resetCache();
        self.speculation.reset(&self.xfm);
        errdefer _ = mlx.mlx_synchronize(self.xfm.s);

        var memory = try MemoryPeak.init();
        memory.start();
        defer memory.stop();

        var forwards: usize = 0;
        self.xfm.benchmark_target_forwards = &forwards;
        defer self.xfm.benchmark_target_forwards = null;

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(a);

        const timer = io_util.Stopwatch.init(self.io);
        var gen = try generate.Generator.initWithOptions(self.io, a, &self.xfm, &self.tok, prompt, limit, sampling, self.config.eosTokenSlice(), options);
        // Upstream's width chooser may otherwise grow back to the checkpoint's
        // full block size after warmup, ignoring a smaller requested depth.
        if (gen.dflash_chooser) |*chooser| {
            chooser.max_width = @min(chooser.max_width, options.dflash_block_size - 1);
        }
        // Even an error must drain GPU work before the next request resets it.
        defer {
            _ = mlx.mlx_synchronize(self.xfm.s);
            gen.deinit(a);
        }

        try memory.sample();
        var first_ns: ?u64 = null;
        var last_ns: ?u64 = null;
        var single: [1]u32 = undefined;
        generation: while (try decode.next(&gen, a, &single)) |batch| {
            defer if (batch.owned) a.free(batch.tokens);
            const now = timer.read();
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
        try memory.sample();
        // DSpark verifies directly in deepseek_v4.zig, bypassing forwardWith.
        // Each completed round has one target verify; rollback uses saved state.
        forwards += @intCast(gen.dspark_attempted);
        gen.logSpecStats();

        const ttft = request_utils.seconds(first_ns orelse finished_ns);
        const decode_seconds = if (first_ns) |first| request_utils.seconds(last_ns.? - first) else 0;
        return .{
            .text = try self.tok.decode(a, ids.items, self.tok.tok_type == .sentencepiece_bpe),
            .tokens_count = ids.items.len,
            .time_to_first_token = ttft,
            .prompt_tps = request_utils.rate(prompt.len, ttft),
            .decode_tps = request_utils.rate(ids.items.len -| 1, decode_seconds),
            .tokens_per_forward_pass = request_utils.rate(ids.items.len, @floatFromInt(forwards)),
            .duration = request_utils.seconds(finished_ns),
            .memory_phys_footprint = memory.snapshot.phys_footprint,
            .memory_resident = memory.snapshot.resident_size,
            .memory_graphics_total = memory.snapshot.graphics_total,
        };
    }
};

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
    const a = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer args.deinit();

    _ = args.next();
    var model_argument: ?[]const u8 = null;
    var draft_argument: ?[]const u8 = null;
    while (args.next()) |flag| {
        if (std.mem.eql(u8, flag, "--help") or std.mem.eql(u8, flag, "-h")) {
            std.debug.print("Usage: engine-mlxserve -m MODEL_DIRECTORY [-d DRAFT_DIRECTORY]\n" ++
                "  -m, --model        Target checkpoint; supported MTP heads auto-enable.\n" ++
                "  -d, --draft-model  DFlash or Gemma assistant checkpoint (overrides MTP).\n" ++
                "Reads BenchRequest JSON Lines and writes BenchResponse arrays.\n" ++
                "speculative_depth: omitted = default, 0 = AR, positive = draft depth.\n", .{});
            return;
        }
        if (std.mem.eql(u8, flag, "-m") or std.mem.eql(u8, flag, "--model")) {
            if (model_argument != null) {
                return error.DuplicateModel;
            }
            model_argument = args.next() orelse return error.ExpectedModel;
        } else if (std.mem.eql(u8, flag, "-d") or std.mem.eql(u8, flag, "--draft-model")) {
            if (draft_argument != null) {
                return error.DuplicateDraftModel;
            }
            draft_argument = args.next() orelse return error.ExpectedDraftModel;
        } else {
            return error.UnexpectedArgument;
        }
    }

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, model_argument orelse return error.ExpectedModel, a);
    defer a.free(path);

    const draft_path = if (draft_argument) |p| try std.Io.Dir.cwd().realPathFileAlloc(io, p, a) else null;
    defer if (draft_path) |p| a.free(p);

    const output_fd = dup(1);
    if (output_fd < 0) {
        return error.StdoutRedirectFailed;
    }
    const output: std.Io.File = .{ .handle = output_fd, .flags = std.Io.File.stdout().flags };
    defer output.close(io);

    if (dup2(2, 1) < 0) {
        return error.StdoutRedirectFailed;
    }

    mlx.installErrorHandler();
    @import("server.zig").applyMlxCacheLimit();
    @import("server.zig").applyGpuCeilingEnv();
    transformer.warmQsaEnvCaches();
    @import("prefix_cache.zig").warmEnvCaches();
    const device = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(device);

    try mlx.check(mlx.mlx_set_default_device(device));
    const engine = try Engine.init(io, a, path, draft_path);
    defer engine.deinit();

    log.info("mlx-serve benchmark ready (v26.10.1, native MLX, mode={s}, depth={d}, max_depth={d})\n", .{
        @tagName(engine.speculation.mode), engine.speculation.default_depth, engine.speculation.max_depth,
    });

    // Allow long prompts while bounding each JSONL record at 64 MiB.
    // Stdin reading and JSON work happen outside runOne's timer.
    const input_buffer = try a.alloc(u8, 64 * 1024 * 1024);
    defer a.free(input_buffer);

    var input = std.Io.File.stdin().reader(io, input_buffer);
    var output_buffer: [16 * 1024]u8 = undefined;
    var writer = output.writer(io, &output_buffer);
    while (try input.interface.takeDelimiter('\n')) |line| {
        if (std.mem.trim(u8, line, " \t\r").len == 0) {
            continue;
        }
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();

        const request = std.json.parseFromSlice(bench.BenchRequest, arena.allocator(), line, .{ .ignore_unknown_fields = true }) catch |err| {
            log.err("Failed to process request: {s}\n", .{@errorName(err)});
            continue;
        };

        const responses = engine.execute(arena.allocator(), request.value) catch |err| {
            log.err("Failed to process request: {s}\n", .{@errorName(err)});
            continue;
        };

        try std.json.Stringify.value(responses, .{}, &writer.interface);
        try writer.interface.writeByte('\n');
        try writer.interface.flush();
    }
}
