const c = @import("memory_c");

// Native inference is single-threaded. The chunk-prefill hook and decode loop
// sample the same request-owned accumulator, without extra GPU synchronization.
var active: ?*MemoryPeak = null;

pub const MemoryPeak = struct {
    snapshot: c.bench_memory_snapshot,

    pub fn init() !MemoryPeak {
        return .{ .snapshot = try collect() };
    }

    pub fn start(self: *MemoryPeak) void {
        active = self;
    }

    pub fn stop(_: *MemoryPeak) void {
        active = null;
    }

    pub fn sample(self: *MemoryPeak) !void {
        const counters = try collect();
        if (counters.graphics_total > self.snapshot.graphics_total) self.snapshot = counters;
    }
};

fn collect() !c.bench_memory_snapshot {
    var counters: c.bench_memory_snapshot = undefined;
    if (c.bench_memory_collect(&counters) != 0) return error.MemoryCountersUnavailable;
    return counters;
}

pub fn sampleActive() !void {
    if (active) |peak| try peak.sample();
}
