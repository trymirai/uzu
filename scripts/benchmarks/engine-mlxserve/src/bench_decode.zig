const std = @import("std");
const generate = @import("generate.zig");
const mtp = @import("mtp.zig");
const drafter = @import("drafter.zig");
const dflash = @import("dflash.zig");
const transformer = @import("transformer.zig");
const Transformer = @import("transformer.zig").Transformer;
const request_utils = @import("bench_request.zig");

pub const Speculation = struct {
    mode: enum { ar, mtp, drafter, dflash, dspark } = .ar,
    mtp_head: ?mtp.MtpModel = null,
    assistant: ?drafter.DrafterModel = null,
    block_drafter: ?dflash.DflashModel = null,
    default_depth: u32 = 0,
    max_depth: u32 = 0,

    // The owner keeps this struct and the target at stable addresses.
    pub fn init(
        self: *Speculation,
        io: std.Io,
        a: std.mem.Allocator,
        xfm: *Transformer,
        model_path: []const u8,
        draft_path: ?[]const u8,
    ) !void {
        self.* = .{};
        errdefer self.deinit();

        if (draft_path) |path| {
            if (dflash.probeIsDflash(io, a, path)) {
                self.block_drafter = try dflash.loadDflash(io, a, xfm.s, path);
                try self.block_drafter.?.bind(xfm);
                const block = self.block_drafter.?.config.block_size;
                if (block < 2) {
                    return error.InvalidDraftBlockSize;
                }

                // Match upstream's loader: older GPUs use a smaller verify
                // block; the checkpoint's full width is not the default there.
                const tree = self.block_drafter.?.selector != null and xfm.specTreeSupported();
                const cap = dflash.blockCapForMachine(@import("ane.zig").chipBrand(), tree);
                const default_block = if (tree and transformer.naxAvailable())
                    dflash.TREE_NAX_BLOCK
                else
                    dflash.resolveBlockSize(block, 0, false, dflash.wideVerifyLaneAvailable(), cap.cap);
                self.mode = .dflash;
                self.default_depth = default_block - 1;
                self.max_depth = @max(block, default_block) - 1;
            } else {
                self.assistant = try drafter.loadDrafter(io, a, xfm.s, path);
                try self.assistant.?.bind(xfm);
                self.mode = .drafter;
                self.default_depth = drafter.recommendedBlockSize(&xfm.config) - 1;
                self.max_depth = std.math.maxInt(u32) - 1;
            }
            return;
        }

        if (mtp.hasMtpHead(io, a, model_path)) {
            self.mtp_head = try mtp.loadMtp(io, a, xfm.s, model_path);
            // A detected but incompatible head is an error, not an AR benchmark.
            try self.mtp_head.?.bind(xfm);
        }
        if (self.head(xfm)) |h| {
            self.mode = .mtp;
            self.default_depth = generate.Generator.resolveMtpDepthCapForProfile(xfm.config.mtpDepth(0), h.costProfile(xfm, xfm.cache.config));
            self.max_depth = mtp.MAX_DEPTH;
        } else if (xfm.dsv4) |m| {
            if (m.n_mtp > 0 and m.ds_block > 0) {
                self.mode = .dspark;
                self.default_depth = @intCast(m.ds_block);
                self.max_depth = @intCast(m.ds_block);
            }
        }
    }

    pub fn deinit(self: *Speculation) void {
        if (self.block_drafter) |*d| d.deinit();
        if (self.assistant) |*d| d.deinit();
        if (self.mtp_head) |*h| h.deinit();
    }

    fn head(self: *Speculation, xfm: *Transformer) ?generate.MtpHeadRef {
        if (self.mtp_head) |*h| return .{ .qwen = h };
        if (xfm.qwen4_mtp != null) return .{ .qwen4 = xfm };
        return null;
    }

    pub fn options(self: *Speculation, xfm: *Transformer, requested: ?usize) !generate.Generator.InitOptions {
        const depth = try request_utils.speculativeDepth(requested, self.default_depth, self.max_depth);
        var out: generate.Generator.InitOptions = .{ .model_has_mtp = self.head(xfm) != null };
        if (depth == 0) {
            return out;
        }

        switch (self.mode) {
            .ar => unreachable,
            .mtp => {
                out.mtp_enabled = true;
                out.mtp = self.head(xfm);
                out.mtp_depth = depth;
                xfm.mtp_depth_free = if (requested != null) depth else generate.Generator.mtpDepthCapFree(xfm.config.mtpDepth(0));
            },
            .drafter => {
                out.drafter_enabled = true;
                out.drafter = &self.assistant.?;
                out.drafter_block_size = depth + 1;
            },
            .dflash => {
                out.dflash_enabled = true;
                out.dflash = &self.block_drafter.?;
                // Depth counts proposals; the block also contains the target anchor.
                out.dflash_block_size = depth + 1;
                // Match the scheduler's width scaling of the block-16 calibration.
                out.dflash_min_accepted_per_round = generate.Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND * @as(f32, @floatFromInt(depth)) / 15.0;
                if (xfm.config.isMoe()) {
                    out.dflash_min_accepted_per_round = @max(out.dflash_min_accepted_per_round, generate.Generator.DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND);
                }
            },
            .dspark => {
                xfm.dsv4.?.ds_block = depth;
                // Upstream switches this to its native DSpark path at init.
                out.mtp_enabled = true;
            },
        }
        return out;
    }

    pub fn reset(self: *Speculation, xfm: *Transformer) void {
        // Each benchmark run starts without timing/acceptance history from prior runs.
        xfm.round_cost = .{ .layout = @import("round_cost.zig").layoutFor(&xfm.config) };
        if (self.mtp_head) |*h| h.ev_seed_accept = null;
        if (xfm.qwen4_mtp) |*h| h.ev_seed_accept = null;
    }
};

pub const Batch = struct {
    tokens: []const u32,
    owned: bool,
};

pub fn next(gen: *generate.Generator, a: std.mem.Allocator, single: *[1]u32) !?Batch {
    const speculative = gen.dspark_enabled or gen.mtp != null or gen.dflash != null or gen.drafter != null;
    if (speculative) {
        const result = (if (gen.dspark_enabled)
            try gen.nextDspark(a)
        else if (gen.mtp != null)
            try gen.nextMtp(a)
        else if (gen.dflash != null)
            try gen.nextDflash(a)
        else
            try gen.nextDrafter(a)) orelse return null;
        return .{ .tokens = result.tokens, .owned = true };
    }
    single[0] = (try gen.next(a)) orelse return null;
    return .{ .tokens = single, .owned = false };
}
