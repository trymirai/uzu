const std = @import("std");

const upstream = "deps/mlx-serve/";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{ .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } } },
    });
    const optimize = b.standardOptimizeOption(.{});
    const ios = target.result.os.tag == .ios;
    const mlx_dir = b.pathResolve(&.{
        b.root.root_dir.path orelse ".",
        b.option([]const u8, "mlx-dir", "MLX and mlx-c install prefix") orelse upstream ++ "lib/mlx",
    });
    const sdk = std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", if (ios) "iphoneos" else "macosx", "--show-sdk-path" }), " \r\n");
    const options = b.addOptions();
    options.addOption([]const u8, "version", "26.10.1");
    options.addOption([]const u8, "git_sha", "02bee553f48cd3bc7d82aba0f8073820bd924738");
    options.addOption([]const u8, "mlx_c_version", "source");
    options.addOption([]const u8, "ds4_commit", "disabled");
    options.addOption([]const u8, "llama_tag", "disabled");
    options.addOption(bool, "macos_engines", false);
    options.addOption(bool, "ios", ios);
    options.addOption(bool, "mas", ios);
    options.addOption(bool, "slow_tests", false);

    const mod = b.createModule(.{
        .root_source_file = b.path(if (ios) upstream ++ "src/bench_ios.zig" else upstream ++ "src/bench_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    mod.addOptions("build_options", options);

    inline for (.{ .{ "mlx_serve_gguf", "lib/mlx-serve-gguf/src/root.zig" }, .{ "sushi_exl3", "lib/sushi/src/exl3/root.zig" } }) |spec| {
        const dependency = b.createModule(.{
            .root_source_file = b.path(upstream ++ spec[1]),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "mlx_host", .module = mod }},
        });
        mod.addImport(spec[0], dependency);
    }
    mod.addImport("mlx_steel_sources", b.createModule(.{
        .root_source_file = b.path(upstream ++ "lib/mlx_steel_sources.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const jinja = b.addTranslateC(.{
        .root_source_file = b.path(upstream ++ "lib/jinja_cpp/jinja_wrapper.h"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("jinja_c", jinja.createModule());
    mod.addIncludePath(b.path(upstream ++ "lib/jinja_cpp"));

    // Compile the vendored Jinja sources instead of using its checked-in archive.
    inline for (.{ "jinja_wrapper", "caps", "lexer", "parser", "runtime", "jinja_string", "value" }) |name| {
        mod.addCSourceFile(.{
            .file = b.path(upstream ++ "lib/jinja_cpp/" ++ name ++ ".cpp"),
            .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" },
        });
    }
    inline for (.{ "ane_bridge", "ane_mlp" }) |name| {
        mod.addCSourceFile(.{
            .file = b.path(upstream ++ "lib/ane/" ++ name ++ ".m"),
            .flags = &.{ "-O3", "-fobjc-arc", "-Wno-deprecated-declarations" },
        });
    }
    mod.addIncludePath(b.path(upstream ++ "lib/ane"));

    const memory = b.addTranslateC(.{
        .root_source_file = b.path("src/bench_memory.h"),
        .target = target,
        .optimize = optimize,
    });
    if (ios) {
        const include: std.Build.LazyPath = .{ .cwd_relative = b.fmt("{s}/usr/include", .{sdk}) };
        jinja.addSystemIncludePath(include);
        memory.addSystemIncludePath(include);
        mod.addSystemIncludePath(include);
        const surface = b.addWriteFiles();
        _ = surface.add("IOSurface/IOSurface.h", "#include <IOSurface/IOSurfaceRef.h>\n");
        mod.addIncludePath(surface.getDirectory());
    }
    mod.addImport("memory_c", memory.createModule());
    mod.addCSourceFile(.{ .file = b.path("src/bench_memory.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{
        .file = b.path("../common-cpp/src/memory_counters.c"),
        .flags = &.{"-O2"},
    });
    mod.addIncludePath(b.path("../common-cpp/src"));
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ mlx_dir, "include" }) });
    mod.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
    if (ios) {
        const lib = b.addLibrary(.{ .name = "engine-mlxserve", .root_module = mod, .linkage = .static });
        lib.bundle_compiler_rt = true;
        b.installArtifact(lib);
    } else {
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ mlx_dir, "lib" }) });
        mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ mlx_dir, "lib" }) });
        mod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
        inline for (.{ "IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface", "Accelerate" }) |name| {
            mod.linkFramework(name, .{});
        }
        const exe = b.addExecutable(.{ .name = "engine-mlxserve", .root_module = mod });
        b.installArtifact(exe);
    }

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_request.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const test_step = b.step("test", "Test benchmark request validation and metrics");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);
}
