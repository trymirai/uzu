const std = @import("std");

pub fn build(b: *std.Build) void {
    _ = @import("build/plugins.zig").generate(b);
    const target = b.standardTargetOptions(.{
        .default_target = .{ .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } } },
    });
    const optimize = b.standardOptimizeOption(.{});
    const ios = target.result.os.tag == .ios;
    const mlx_dir = b.pathResolve(&.{
        b.root.root_dir.path orelse ".",
        b.option([]const u8, "mlx-dir", "MLX and mlx-c install prefix") orelse "lib/mlx",
    });
    const sdk = std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", if (ios) "iphoneos" else "macosx", "--show-sdk-path" }), " \r\n");
    const common_dir = b.option([]const u8, "common-dir", "Shared benchmark sources") orelse "../../../common-cpp/src";
    const options = b.addOptions();
    options.addOption([]const u8, "version", @import("build/config.zig").readAppVersion(b) orelse @panic("Missing upstream version"));
    options.addOption([]const u8, "git_sha", "");
    options.addOption([]const u8, "mlx_c_version", "source");
    options.addOption([]const u8, "ds4_commit", "disabled");
    options.addOption([]const u8, "llama_tag", "disabled");
    options.addOption(bool, "ds4_engine", false);
    options.addOption(bool, "llama_engine", false);
    options.addOption(bool, "mlx_stream", false);
    options.addOption(bool, "units", false);
    options.addOption(bool, "plugin_profile", false);
    options.addOption(bool, "ios", ios);
    options.addOption(bool, "mas", ios);
    options.addOption(bool, "slow_tests", false);

    const mod = b.createModule(.{
        .root_source_file = b.path(if (ios) "src/bench_ios.zig" else "src/bench_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    mod.addOptions("build_options", options);
    const libs = @import("build/libs.zig");
    libs.addVendoredSources(b, mod);
    mod.addImport("stb", libs.addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, sdk));
    const webp_include = std.mem.trim(u8, b.run(&.{ "pkg-config", "--variable=includedir", "libwebp" }), " \r\n");
    mod.addImport("webp", libs.addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{webp_include}) }, .{ .cwd_relative = webp_include }, target, optimize, sdk));

    mod.addImport("sushi_exl3", b.createModule(.{
        .root_source_file = b.path("src/sdk/quant/exl3/sushi/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "mlx_host", .module = mod }},
    }));
    mod.addImport("mlx_steel_sources", b.createModule(.{
        .root_source_file = b.path("lib/mlx_steel_sources.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const jinja = b.addTranslateC(.{
        .root_source_file = b.path("lib/jinja_cpp/jinja_wrapper.h"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("jinja_c", jinja.createModule());
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // Compile the vendored Jinja sources instead of using its checked-in archive.
    inline for (.{ "jinja_wrapper", "caps", "lexer", "parser", "runtime", "jinja_string", "value" }) |name| {
        mod.addCSourceFile(.{
            .file = b.path("lib/jinja_cpp/" ++ name ++ ".cpp"),
            .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" },
        });
    }
    inline for (.{ "ane_bridge", "ane_mlp" }) |name| {
        mod.addCSourceFile(.{
            .file = b.path("lib/ane/" ++ name ++ ".m"),
            .flags = &.{ "-O3", "-fobjc-arc", "-Wno-deprecated-declarations" },
        });
    }
    mod.addIncludePath(b.path("lib/ane"));

    const system = b.addTranslateC(.{
        .root_source_file = b.path("src/bench_system.h"),
        .target = target,
        .optimize = optimize,
    });
    const include: std.Build.LazyPath = .{ .cwd_relative = b.fmt("{s}/usr/include", .{sdk}) };
    jinja.addSystemIncludePath(include);
    system.addSystemIncludePath(include);
    system.addIncludePath(.{ .cwd_relative = b.pathResolve(&.{common_dir}) });
    mod.addSystemIncludePath(include);
    mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/usr/lib", .{sdk}) });
    if (ios) {
        const surface = b.addWriteFiles();
        _ = surface.add("IOSurface/IOSurface.h", "#include <IOSurface/IOSurfaceRef.h>\n");
        mod.addIncludePath(surface.getDirectory());
    }
    mod.addImport("system_c", system.createModule());
    mod.addCSourceFile(.{
        .file = .{ .cwd_relative = b.pathResolve(&.{ common_dir, "memory_counters.c" }) },
        .flags = &.{"-O2"},
    });
    mod.addCSourceFile(.{
        .file = .{ .cwd_relative = b.pathResolve(&.{ common_dir, "apple_temp_sensors.c" }) },
        .flags = &.{"-O2"},
    });
    mod.addCSourceFile(.{
        .file = .{ .cwd_relative = b.pathResolve(&.{ common_dir, "energy_counters.c" }) },
        .flags = &.{"-O2"},
    });
    mod.addIncludePath(.{ .cwd_relative = b.pathResolve(&.{common_dir}) });
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ mlx_dir, "include" }) });
    mod.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
    if (ios) {
        const lib = b.addLibrary(.{ .name = "engine-mlxserve", .root_module = mod, .linkage = .static });
        lib.bundle_compiler_rt = true;
        b.installArtifact(lib);
    } else {
        mod.linkSystemLibrary("webp", .{ .use_pkg_config = .force });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ mlx_dir, "lib" }) });
        mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ mlx_dir, "lib" }) });
        mod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
        mod.linkSystemLibrary("IOReport", .{ .use_pkg_config = .no });
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
