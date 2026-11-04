const std = @import("std");
const bench_main = @import("bench_main.zig");
pub const mlx = bench_main.mlx;
pub const log = bench_main.log;
pub const io_util = bench_main.io_util;

export fn benchmarks_mlxserve_main(argc: c_int, argv: [*][*:0]u8) c_int {
    var count: usize = 0;
    while (std.c.environ[count] != null) : (count += 1) {}
    const environ: std.process.Environ.Block = .{ .slice = std.c.environ[0..count :null] };
    const args = argv[0..@intCast(argc)];
    const gpa = std.heap.c_allocator;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var threaded: std.Io.Threaded = .init(gpa, .{ .argv0 = .init(.{ .vector = args }), .environ = .{ .block = environ } });
    defer threaded.deinit();
    var environ_map = std.process.Environ.createMap(.{ .block = environ }, gpa) catch return 1;
    defer environ_map.deinit();
    const preopens = std.process.Preopens.init(arena.allocator()) catch return 1;

    bench_main.main(.{
        .minimal = .{ .args = .{ .vector = args }, .environ = .{ .block = environ } },
        .arena = &arena,
        .gpa = gpa,
        .io = threaded.io(),
        .environ_map = &environ_map,
        .preopens = preopens,
    }) catch |err| {
        log.err("{s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
